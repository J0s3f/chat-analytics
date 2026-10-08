import { strToU8, zipSync } from "fflate";

import { Env } from "@pipeline/Env";
import { LanguageCodes } from "@pipeline/Languages";
import { Emojis } from "@pipeline/process/nlp/Emojis";
import { Sentiment } from "@pipeline/process/nlp/Sentiment";
import { tokenize } from "@pipeline/process/nlp/Tokenizer";

import { TestEnv } from "@tests/samples";

describe("Sentiment", () => {
    let sentiment: Sentiment;

    beforeAll(async () => {
        sentiment = await Sentiment.load(TestEnv, await Emojis.load(TestEnv));
    });

    it("should detect sentiment in basic sentences", () => {
        expect(sentiment.calculate(tokenize("i love you"), LanguageCodes.indexOf("en"))).toBePositive();
        expect(sentiment.calculate(tokenize("i hate you"), LanguageCodes.indexOf("en"))).toBeNegative();
    });

    it("should detect sentiment in emojis", () => {
        expect(sentiment.calculate(tokenize("💓"), LanguageCodes.indexOf("en"))).toBePositive();
        expect(sentiment.calculate(tokenize("😠"), LanguageCodes.indexOf("en"))).toBeNegative();
    });

    test.each(["gut", "Gut", "GUT", "GuT"])("matches German casing consistently: %s", (text) => {
        expect(sentiment.calculate(tokenize(text), LanguageCodes.indexOf("de"))).toBe(3);
    });

    test.each([
        ["ned guad", -3],
        ["NET GUAT", -3],
        ["nit schlecht", 2],
        ["kein gut", -3],
        ["leiwand", 3],
        ["schee", 3],
        ["nicht gut aber schön", 0],
        ["ned guad owa schön", 0],
        ["nicht gut. schön", 0],
        ["nicht nur gut sondern schön", 6],
    ])("handles German negation and spelling: %s", (text, expected) => {
        expect(sentiment.calculate(tokenize(text), LanguageCodes.indexOf("de"))).toBe(expected);
    });

    it("distinguishes no recognized sentiment from balanced sentiment", () => {
        expect(sentiment.calculate(tokenize("oida"), LanguageCodes.indexOf("de"))).toBeUndefined();
        expect(sentiment.calculate(tokenize("gut schön schlecht"), 0)).toBeUndefined();
    });

    it("matches each longest phrase once and does not mutate tokens", async () => {
        const zip = zipSync({
            "AFINN-en.json": strToU8(JSON.stringify({ good: 3, "very good": 4, fun: 3, "no fun": -3 })),
            "all-negators.json": strToU8(JSON.stringify({ en: ["not", "was not"] })),
        });
        const env: Env = { loadAsset: jest.fn().mockResolvedValue(zip.buffer) };
        const custom = await Sentiment.load(env, await Emojis.load(TestEnv));
        const lang = LanguageCodes.indexOf("en");
        const tokens = tokenize("Very GOOD");
        const original = tokens.map((token) => ({ ...token }));
        expect(custom.calculate(tokens, lang)).toBe(4);
        expect(tokens).toEqual(original);
        expect(custom.calculate(tokenize("No FUN"), lang)).toBe(-3);
        expect(custom.calculate(tokenize("was not good"), lang)).toBe(-3);
        expect(custom.calculate(tokenize("not good but fun"), lang)).toBe(0);
        expect(custom.calculate(tokenize("not only good but fun"), lang)).toBe(6);
    });
});
