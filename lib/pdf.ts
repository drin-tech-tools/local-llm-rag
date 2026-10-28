// Import the inner module to avoid pdf-parse's debug self-test on load.
// @ts-ignore -- inner module path has no type declarations
import pdfParse from "pdf-parse/lib/pdf-parse.js";

export async function extractPages(buf: Buffer): Promise<string[]> {
  const pages: string[] = [];
  await pdfParse(buf, {
    pagerender: async (pageData: any) => {
      const tc = await pageData.getTextContent();
      const text = tc.items.map((i: any) => i.str).join(" ");
      pages[pageData.pageIndex] = text;
      return text;
    },
  });
  return pages.map((p) => p ?? "");
}
