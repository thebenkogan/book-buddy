import {
  Book,
  BookContent,
  AISummary,
  TableOfContents,
  AskResponse,
} from "@/types/types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export const getBooks = async (): Promise<Book[]> => {
  const response = await fetch(`${API_BASE_URL}/books`);
  if (!response.ok) {
    throw new Error("Failed to fetch books");
  }
  return response.json();
};

export const getBookContent = async (bookId: string): Promise<BookContent> => {
  const response = await fetch(`${API_BASE_URL}/books/${bookId}/content`);
  if (!response.ok) {
    throw new Error(`Failed to fetch content for book: ${bookId}`);
  }
  return response.json();
};

export const getTableOfContents = async (
  bookId: string,
): Promise<TableOfContents> => {
  const response = await fetch(`${API_BASE_URL}/books/${bookId}/toc`);
  if (!response.ok) {
    throw new Error(`Failed to fetch TOC for book: ${bookId}`);
  }
  return response.json();
};

export const getAISummary = async (bookId: string): Promise<AISummary> => {
  const response = await fetch(`${API_BASE_URL}/books/${bookId}/summary`);
  if (!response.ok) {
    throw new Error(`Failed to fetch summary for book: ${bookId}`);
  }
  return response.json();
};

export const askQuestion = async (
  bookId: string,
  question: string,
): Promise<string> => {
  const response = await fetch(`${API_BASE_URL}/books/${bookId}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    throw new Error(`Failed to ask question for book: ${bookId}`);
  }
  const data: AskResponse = await response.json();
  return data.answer;
};

export interface ReadingProgressPayload {
  userId: string;
  currentChapter: number;
}

export const updateReadingProgress = async (
  bookId: string,
  payload: ReadingProgressPayload,
): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/books/${bookId}/progress`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: payload.userId,
      current_chapter: payload.currentChapter,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to update reading progress for book: ${bookId}`);
  }
};
