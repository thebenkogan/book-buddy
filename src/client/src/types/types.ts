export interface Book {
  id: string;
  title: string;
  author?: string;
  coverImage?: string;
  progress: number;
  lastRead?: Date;
}

export interface SearchBook {
  book_id: string;
  title: string;
  author: string;
  download_count: number;
  languages: string[];
  indexed: boolean;
  cover_url?: string;
}

export interface SearchResponse {
  count: number;
  results: SearchBook[];
}

export interface BookContent {
  text: string;
  currentPosition: number;
}

export interface TOCChapter {
  id: number;
  name: string;
  context?: {
    [rank: number]: string;
  };
  startPosition: number;
  endPosition?: number;
}

export interface TableOfContents {
  chapters: TOCChapter[];
}

export interface AISummary {
  summary: string;
  keyPoints: string[];
}

export interface AskResponse {
  answer: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}
