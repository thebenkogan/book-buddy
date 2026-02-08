import { Book, BookContent, AISummary } from '@/types/types';

// Mock data for development - replace with actual API calls to your Python backend
const mockBooks: Book[] = [
  {
    id: '1',
    title: 'The Great Gatsby',
    author: 'F. Scott Fitzgerald',
    progress: 45,
    lastRead: new Date('2024-01-15'),
  },
  {
    id: '2',
    title: 'To Kill a Mockingbird',
    author: 'Harper Lee',
    progress: 78,
    lastRead: new Date('2024-01-20'),
  },
  {
    id: '3',
    title: '1984',
    author: 'George Orwell',
    progress: 23,
    lastRead: new Date('2024-01-10'),
  },
];

const mockContent = `Chapter 1: The Beginning

In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since.

"Whenever you feel like criticizing any one," he told me, "just remember that all the people in this world haven't had the advantages that you've had."

He didn't say any more, but we've always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that. In consequence, I'm inclined to reserve all judgments, a habit that has opened up many curious natures to me and also made me the victim of not a few veteran bores.

The abnormal mind is quick to detect and attach itself to this quality when it appears in a normal person, and so it came about that in college I was unjustly accused of being a politician, because I was privy to the secret griefs of wild, unknown men.

Most of the confidences were unsought — frequently I have feigned sleep, preoccupation, or a hostile levity when I realized by some unmistakable sign that an intimate revelation was quivering on the horizon; for the intimate revelations of young men, or at least the terms in which they express them, are usually plagiaristic and marred by obvious suppressions.`;

// API functions - replace URLs with your actual Python API endpoints
const API_BASE_URL = 'http://localhost:8000/api'; // Update with your API URL

export const getBooks = async (): Promise<Book[]> => {
  // TODO: Replace with actual API call
  // const response = await fetch(`${API_BASE_URL}/books`);
  // return response.json();
  
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockBooks), 300);
  });
};

export const getBookContent = async (bookId: string): Promise<BookContent> => {
  // TODO: Replace with actual API call
  // const response = await fetch(`${API_BASE_URL}/books/${bookId}/content`);
  // return response.json();
  
  return new Promise((resolve) => {
    setTimeout(() => resolve({
      text: mockContent,
      currentPosition: 0,
    }), 300);
  });
};

export const getAISummary = async (bookId: string, section?: string): Promise<AISummary> => {
  // TODO: Replace with actual API call
  // const response = await fetch(`${API_BASE_URL}/books/${bookId}/summary`, {
  //   method: 'POST',
  //   body: JSON.stringify({ section }),
  // });
  // return response.json();
  
  return new Promise((resolve) => {
    setTimeout(() => resolve({
      summary: "This opening chapter introduces the narrator and establishes the moral framework of the story. The narrator reflects on advice from his father about reserving judgment, which has shaped his character and made him a confidant to many people.",
      keyPoints: [
        "The narrator's father advised him to reserve judgment of others",
        "This quality has attracted many people to confide in him",
        "He was incorrectly labeled as a politician in college",
        "The narrator is critical of how young men share their intimate revelations"
      ]
    }), 500);
  });
};

export const askQuestion = async (bookId: string, question: string): Promise<string> => {
  // TODO: Replace with actual API call
  // const response = await fetch(`${API_BASE_URL}/books/${bookId}/ask`, {
  //   method: 'POST',
  //   body: JSON.stringify({ question }),
  // });
  // const data = await response.json();
  // return data.answer;
  
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve("Based on the text, the narrator's father advised him to reserve judgment when criticizing others, reminding him that not everyone has had the same advantages. This advice has significantly influenced the narrator's approach to life and relationships.");
    }, 800);
  });
};

export const updateReadingProgress = async (bookId: string, progress: number): Promise<void> => {
  // TODO: Replace with actual API call
  // await fetch(`${API_BASE_URL}/books/${bookId}/progress`, {
  //   method: 'PUT',
  //   body: JSON.stringify({ progress }),
  // });
  
  console.log(`Updating progress for book ${bookId}: ${progress}%`);
};
