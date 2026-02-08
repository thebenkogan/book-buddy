import { useQuery } from '@tanstack/react-query';
import { getTableOfContents } from '@/services/bookService';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { List, ChevronRight } from 'lucide-react';
import { Chapter } from '@/types/types';
import { cn } from '@/lib/utils';

interface TableOfContentsProps {
  bookId: string;
  currentPosition?: number;
  onChapterClick?: (chapter: Chapter) => void;
}

interface GroupedChapters {
  [key: string]: Chapter[];
}

const TableOfContents = ({ bookId, currentPosition = 0, onChapterClick }: TableOfContentsProps) => {
  const { data: toc, isLoading } = useQuery({
    queryKey: ['tableOfContents', bookId],
    queryFn: () => getTableOfContents(bookId),
  });

  // Group chapters by their hierarchical context
  const groupedChapters = toc?.chapters.reduce<GroupedChapters>((acc, chapter) => {
    const contextKeys = Object.keys(chapter.context || {}).sort((a, b) => Number(a) - Number(b));
    const groupKey = contextKeys.map(key => chapter.context![Number(key)]).join(' > ') || 'Chapters';
    
    if (!acc[groupKey]) {
      acc[groupKey] = [];
    }
    acc[groupKey].push(chapter);
    return acc;
  }, {});

  const isChapterActive = (chapter: Chapter) => {
    return currentPosition >= chapter.startPosition && currentPosition <= chapter.endPosition;
  };

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <Skeleton className="h-6 w-32" />
        </CardHeader>
        <CardContent className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-4/5" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="border-b">
        <CardTitle className="flex items-center gap-2 text-lg">
          <List size={20} />
          Table of Contents
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 p-0 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-4 space-y-6">
            {Object.entries(groupedChapters || {}).map(([groupKey, chapters]) => (
              <div key={groupKey} className="space-y-2">
                {/* Section Header */}
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2">
                  {groupKey}
                </div>
                
                {/* Chapters in this section */}
                <div className="space-y-1">
                  {chapters.map((chapter) => {
                    const isActive = isChapterActive(chapter);
                    return (
                      <button
                        key={chapter.id}
                        onClick={() => onChapterClick?.(chapter)}
                        className={cn(
                          "w-full text-left px-3 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 group",
                          isActive
                            ? "bg-blue-100 text-blue-900 font-medium shadow-sm"
                            : "hover:bg-gray-100 text-gray-700"
                        )}
                      >
                        <ChevronRight 
                          size={16} 
                          className={cn(
                            "flex-shrink-0 transition-transform",
                            isActive ? "text-blue-600" : "text-gray-400 group-hover:text-gray-600"
                          )}
                        />
                        <span className="text-sm flex-1 line-clamp-2">
                          {chapter.name}
                        </span>
                        {isActive && (
                          <div className="w-2 h-2 rounded-full bg-blue-600 animate-pulse flex-shrink-0" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default TableOfContents;
