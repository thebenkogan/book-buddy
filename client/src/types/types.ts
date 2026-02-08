export interface Book {
  id: string;
  title: string;
  author?: string;
  coverImage?: string;
  progress: number;
  lastRead?: Date;
}

export interface BookContent {
  text: string;
  currentPosition: number;
}

export interface AISummary {
  summary: string;
  keyPoints: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
