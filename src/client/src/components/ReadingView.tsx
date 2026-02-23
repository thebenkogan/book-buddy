import { useQuery } from '@tanstack/react-query';
import { getBookContent } from '@/services/bookService';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { useEffect, useRef, useState } from 'react';
import { Chapter } from '@/types/types';

interface ReadingViewProps {
  bookId: string;
  onPositionChange?: (position: number) => void;
  scrollToChapter?: Chapter | null;
}

const ReadingView = ({ bookId, onPositionChange, scrollToChapter }: ReadingViewProps) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [contentLength, setContentLength] = useState(0);
  
  const { data: content, isLoading } = useQuery({
    queryKey: ['bookContent', bookId],
    queryFn: () => getBookContent(bookId),
  });

  useEffect(() => {
    if (content?.text) {
      setContentLength(content.text.length);
    }
  }, [content]);

  // Handle scroll to chapter
  useEffect(() => {
    if (scrollToChapter && scrollAreaRef.current) {
      const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (viewport) {
        const scrollPercentage = scrollToChapter.startPosition / contentLength;
        const scrollPosition = viewport.scrollHeight * scrollPercentage;
        viewport.scrollTo({ top: scrollPosition, behavior: 'smooth' });
      }
    }
  }, [scrollToChapter, contentLength]);

  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const target = event.target as HTMLDivElement;
    const scrollPercentage = target.scrollTop / (target.scrollHeight - target.clientHeight);
    const position = Math.floor(scrollPercentage * contentLength);
    onPositionChange?.(position);
  };

  if (isLoading) {
    return (
      <div className="h-full p-8 space-y-4">
        <Skeleton className="h-8 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-4/5" />
      </div>
    );
  }

  return (
    <ScrollArea className="h-full" ref={scrollAreaRef}>
      <div 
        className="max-w-4xl mx-auto px-8 py-12"
        onScroll={handleScroll}
      >
        <article className="prose prose-lg prose-slate max-w-none">
          <div className="whitespace-pre-wrap leading-relaxed text-foreground">
            {content?.text}
          </div>
        </article>
      </div>
    </ScrollArea>
  );
};

export default ReadingView;
