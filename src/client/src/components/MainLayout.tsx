import { useState } from "react";
import { useParams } from "react-router-dom";
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import BookSidebar from "@/components/BookSidebar";
import ReadingView from "@/components/ReadingView";
import AIView from "@/components/AIView";
import TableOfContents from "@/components/TableOfContents";
import { BookOpen } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getBooks } from "@/services/bookService";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { TOCChapter } from "@/types/types";

const MainLayout = () => {
  const { bookId } = useParams();
  const [currentPosition, setCurrentPosition] = useState(0);
  const [scrollToChapter, setScrollToChapter] = useState<TOCChapter | null>(
    null,
  );

  const { data: books } = useQuery({
    queryKey: ["books"],
    queryFn: getBooks,
  });

  const currentBook = books?.find((b) => b.id === bookId);

  const handleChapterClick = (chapter: TOCChapter) => {
    setScrollToChapter(chapter);
    setCurrentPosition(chapter.startPosition);
    // Reset after a short delay to allow for multiple clicks on the same chapter
    setTimeout(() => setScrollToChapter(null), 100);
  };

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
                <h1 className="text-2xl font-bold">
                  {currentBook?.title || "Book"}
                </h1>
                {currentBook?.author && (
                  <p className="text-sm text-muted-foreground">
                    by {currentBook.author}
                  </p>
                )}
              </div>
            </header>

            {/* Resizable Panes */}
            <div className="flex-1 overflow-hidden">
              <ResizablePanelGroup direction="horizontal" className="h-full">
                {/* Table of Contents */}
                <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
                  <div className="h-full p-4 bg-gray-50/50">
                    <TableOfContents
                      bookId={bookId}
                      currentPosition={currentPosition}
                      onChapterClick={handleChapterClick}
                    />
                  </div>
                </ResizablePanel>

                <ResizableHandle withHandle />

                {/* Reading View */}
                <ResizablePanel defaultSize={45} minSize={30}>
                  <div className="h-full bg-white">
                    <ReadingView
                      bookId={bookId}
                      onPositionChange={setCurrentPosition}
                      scrollToChapter={scrollToChapter}
                    />
                  </div>
                </ResizablePanel>

                <ResizableHandle withHandle />

                {/* AI Analysis View */}
                <ResizablePanel defaultSize={35} minSize={25}>
                  <div className="h-full bg-gray-50/30">
                    <AIView bookId={bookId} />
                  </div>
                </ResizablePanel>
              </ResizablePanelGroup>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-screen">
            <div className="text-center space-y-4">
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-6 rounded-full w-24 h-24 mx-auto flex items-center justify-center">
                <BookOpen className="text-white" size={48} />
              </div>
              <h2 className="text-2xl font-bold">Welcome to BookBuddy AI</h2>
              <p className="text-muted-foreground max-w-md">
                Select a book from the sidebar to start reading or analyzing
                with AI
              </p>
            </div>
          </div>
        )}
      </SidebarInset>
    </div>
  );
};

export default MainLayout;
