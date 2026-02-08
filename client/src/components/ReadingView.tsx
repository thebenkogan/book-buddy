import { useQuery } from '@tanstack/react-query';
import { getBookContent } from '@/services/bookService';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';

interface ReadingViewProps {
  bookId: string;
}

const ReadingView = ({ bookId }: ReadingViewProps) => {
  const { data: content, isLoading } = useQuery({
    queryKey: ['bookContent', bookId],
    queryFn: () => getBookContent(bookId),
  });

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
    <ScrollArea className="h-full">
      <div className="max-w-4xl mx-auto px-8 py-12">
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
