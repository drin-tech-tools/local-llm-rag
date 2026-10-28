import "./globals.css";

export const metadata = {
  title: "Local LLM RAG",
  description: "Chat with your PDFs using self-hosted models",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
