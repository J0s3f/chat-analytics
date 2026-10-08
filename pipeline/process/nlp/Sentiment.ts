import { unzipSync } from "fflate";

import { Env } from "@pipeline/Env";
import { Language, LanguageCodes } from "@pipeline/Languages";
import { Progress } from "@pipeline/Progress";
import { Index } from "@pipeline/Types";
import { Emojis } from "@pipeline/process/nlp/Emojis";
import { normalizeText } from "@pipeline/process/nlp/Text";
import { Token, tokenize } from "@pipeline/process/nlp/Tokenizer";

const germanNegators = ["nie", "niemals", "kein", "keine", "keinen", "keinem", "keiner", "ned", "net", "nit"];
const germanAliases: Record<string, string> = {
    guad: "gut",
    guat: "gut",
    guade: "gut",
    guate: "gut",
    schee: "schön",
    leiwand: "gut",
};
const clauseBoundaries: Partial<Record<Language, string[]>> = {
    de: ["aber", "sondern", "jedoch", "doch", "owa"],
    en: ["but", "however", "yet"],
};

export class Sentiment {
    private readonly langs: {
        [lang: Index]: {
            negators: PatternMatcher;
            afinn: PatternMatcher;
            boundaries: Set<string>;
        };
    } = {};

    private constructor(afinnZipBuffer: ArrayBuffer, private emojiData: Emojis, progress?: Progress) {
        const filesAsBuffers = unzipSync(new Uint8Array(afinnZipBuffer));
        const filesAsStrings: { [key: string]: string } = {};

        // transform buffers into strings
        for (const filename in filesAsBuffers) {
            filesAsStrings[filename] = new TextDecoder("utf-8").decode(filesAsBuffers[filename]);
        }

        // read all-negators.json
        const negators = JSON.parse(filesAsStrings["all-negators.json"]) as { [lang: string]: string[] };

        progress?.new("Preparing sentiment database");
        const total = Object.keys(filesAsStrings).length;
        let processed = 0;
        for (const filename in filesAsStrings) {
            if (filename.startsWith("AFINN-")) {
                const lang = filename.substring(6, filename.length - 5) as Language;
                const langIndex = LanguageCodes.indexOf(lang);
                if (langIndex === -1) {
                    // TODO: fixme
                    // console.log("Skipping language", lang);
                    continue;
                }

                const langNegators = [...(negators[lang] ?? []), ...(lang === "de" ? germanNegators : [])];
                const langAfinn = JSON.parse(filesAsStrings[filename]) as { [word: string]: number };

                // Reuse existing dictionary weights for conservative spelling aliases.
                if (lang === "de") {
                    for (const [alias, canonical] of Object.entries(germanAliases)) {
                        if (langAfinn[alias] === undefined && langAfinn[canonical] !== undefined) {
                            langAfinn[alias] = langAfinn[canonical];
                        }
                    }
                }

                this.langs[langIndex] = {
                    negators: new PatternMatcher(langNegators),
                    afinn: new PatternMatcher(Object.keys(langAfinn), Object.values(langAfinn)),
                    boundaries: new Set(clauseBoundaries[lang] ?? []),
                };
            }
            progress?.progress("number", processed++, total);
        }
        progress?.success();
    }

    // NOTE: based on marcellobarile/multilang-sentiment
    calculate(tokens: Token[], lang: Index): number | undefined {
        const langDb = this.langs[lang];
        if (langDb === undefined) return undefined;

        // TODO: RTL

        let score = 0;
        let nearNegator = false;
        let nearNegatorDist = 0;
        let tokenHits = 0;

        for (let i = 0, len = tokens.length; i < len; i++) {
            const token = tokens[i];
            const word = token.text.toLowerCase();

            // non-words reset the negator
            // also if we are too far
            if (token.tag !== "word" || nearNegatorDist > 5 || langDb.boundaries.has(word)) nearNegator = false;
            nearNegatorDist++;

            if (token.tag === "emoji") {
                const emojiValue = this.emojiData.getSentiment(token.text);
                if (emojiValue !== undefined) {
                    tokenHits++;
                    score += emojiValue;
                }
                continue;
            }

            // only handle words
            if (token.tag == "word") {
                // "not only" and "nicht nur" introduce emphasis rather than negative polarity.
                const next = tokens[i + 1];
                if (
                    next?.tag === "word" &&
                    ((LanguageCodes[lang] === "de" && word === "nicht" && next.text.toLowerCase() === "nur") ||
                        (LanguageCodes[lang] === "en" && word === "not" && next.text.toLowerCase() === "only"))
                ) {
                    nearNegator = false;
                    i++;
                    continue;
                }

                const negatorMatch = langDb.negators.match(tokens, i);
                if (negatorMatch) {
                    nearNegator = true;
                    nearNegatorDist = 0;
                    i += negatorMatch.pattern.length;
                    continue;
                }

                const afinnMatch = langDb.afinn.match(tokens, i);
                if (afinnMatch) {
                    tokenHits++;
                    score += afinnMatch.value * (nearNegator ? -1 : 1);
                    // Consume the whole phrase; its suffix must not be counted a second time.
                    i += afinnMatch.pattern.length;
                    nearNegatorDist += afinnMatch.pattern.length;
                }
            }
        }

        // only return score if we found at least one token with sentiment
        return tokenHits > 0 ? score : undefined;
    }

    static async load(env: Env, emojis: Emojis) {
        const afinnZipBuffer = await env.loadAsset("/data/text/AFINN.zip", "arraybuffer");
        return new Sentiment(afinnZipBuffer, emojis, env.progress);
    }
}

interface PatternEntry {
    value: number;
    pattern: string[];
}

// TODO: fuzzing?
class PatternMatcher {
    private readonly mapping: Map<string, PatternEntry[]>;

    constructor(patterns: string[], values?: number[]) {
        if (values) console.assert(patterns.length === values.length);
        const len = patterns.length;

        // normalize patterns
        patterns = patterns.map((pattern) => normalizeText(pattern).toLowerCase());
        // tokenize patterns
        const patternsTokenized = patterns
            .map((pattern) => tokenize(pattern))
            .map((tokens) =>
                tokens.filter((token) => token.tag === "word" || token.tag === "emoji").map((token) => token.text)
            );

        // map patterns
        this.mapping = new Map();
        for (let i = 0; i < len; i++) {
            const pattern = patternsTokenized[i];
            if (pattern.length === 0) continue;
            const entry = { value: values ? values[i] : 1, pattern: pattern.slice(1) };
            if (this.mapping.has(pattern[0])) {
                this.mapping.get(pattern[0])!.push(entry);
            } else {
                this.mapping.set(pattern[0], [entry]);
            }
        }

        // sort patterns by specificity
        for (const key of this.mapping.keys()) {
            const group = this.mapping.get(key);
            group?.sort((a, b) => b.pattern.length - a.pattern.length);
        }
    }

    match(input: Token[], offset = 0): PatternEntry | undefined {
        if (offset >= input.length) return undefined;

        const group = this.mapping.get(input[offset].text.toLowerCase());
        if (group === undefined) return undefined;

        for (const entry of group) {
            const len = entry.pattern.length;
            if (input.length - offset - 1 < len) continue;
            let i = 0;
            while (
                i < len &&
                input[offset + i + 1].tag === "word" &&
                input[offset + i + 1].text.toLowerCase() === entry.pattern[i]
            )
                i++;
            if (i === len) return entry;
        }

        return undefined;
    }
}
