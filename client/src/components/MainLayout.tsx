import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { SidebarInset, SidebarTrigger } from '@/components/ui/sidebar';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import BookSidebar from '@/components/BookSidebar';
import ReadingView from '@/components/ReadingView';
import AIView from '@/components/AIView';
import { BookOpen, Sparkles } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { getBooks } from '@/services/bookService';

const MainLayout = () => {
  const { bookId } = useParams();
  const [activeTab, setActiveTab] = useState('reading');

  const { data: books } = useQuery({
    queryKey: ['books'],
    queryFn: getBooks,
  });

  const currentBook = books?.find(b => b.id === bookId);

  return (
    <div className="flex min-h-screen w-full">
      <BookSidebar />
      
      <SidebarInset className="flex-1 w-full min-w-0">
        {bookId ? (
          <div className="flex flex-col h-screen">
            {/* Header */}
            <header className="flex items-center gap-4 border-b bg-white px-6 py-4 sticky top-0 z-10">
              <SidebarTrigger />
              <div className="flex-1">
                <h1 className="text-2xl font-bold">{currentBook?.title || 'Book'}</h1>
                {currentBook?.author && (
                  <p className="text-sm text-muted-foreground">by {currentBook.author}</p>
                )}
              </div>
            </header>

            {/* Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
              <div className="border-b bg-white px-6">
                <TabsList className="h-12">
                  <TabsTrigger value="reading" className="gap-2">
                    <BookOpen size={18} />
                    Reading View
                  </TabsTrigger>
                  <TabsTrigger value="ai" className="gap-2">
                    <Sparkles size={18} />
                    AI Analysis
                  </TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="reading" className="flex-1 m-0 overflow-hidden">
                <ReadingView bookId={bookId} />
              </TabsContent>

              <TabsContent value="ai" className="flex-1 m-0 overflow-hidden">
                <AIView bookId={bookId} />
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <div className="flex items-center justify-center h-screen">
            <div className="text-center space-y-4">
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-6 rounded-full w-24 h-24 mx-auto flex items-center justify-center">
                <BookOpen className="text-white" size={48} />
              </div>
              <h2 className="text-2xl font-bold">Welcome to BookBuddy AI</h2>
              <p className="text-muted-foreground max-w-md">
                Select a book from the sidebar to start reading or analyzing with AI
              </p>
            </div>
          </div>
        )}
      </SidebarInset>
    </div>
  );
};

export default MainLayout;
