import { useQuery } from "@tanstack/react-query";
import { getBookContent, getTableOfContents } from "@/services/bookService";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { useEffect, useRef, useState, useMemo } from "react";
import { TOCChapter } from "@/types/types";

interface ReadingViewProps {
  bookId: string;
  onPositionChange?: (position: number) => void;
  scrollToChapter?: TOCChapter | null;
}

const ReadingView = ({
  bookId,
  onPositionChange,
  scrollToChapter,
}: ReadingViewProps) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const chapterRefs = useRef<Map<number, HTMLSpanElement>>(new Map());
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { data: content, isLoading: contentLoading } = useQuery({
    queryKey: ["bookContent", bookId],
    queryFn: () => getBookContent(bookId),
  });

  const { data: toc } = useQuery({
    queryKey: ["tableOfContents", bookId],
    queryFn: () => getTableOfContents(bookId),
  });

  const chapters = useMemo(() => {
    if (!toc?.chapters || !content?.text) return [];

    const sorted = [...toc.chapters].sort(
      (a, b) => a.startPosition - b.startPosition,
    );
    return sorted.map((chapter, index) => {
      const nextChapter = sorted[index + 1];
      const endPosition = nextChapter
        ? nextChapter.startPosition
        : content.text.length;
      return {
        ...chapter,
        endPosition,
        text: content.text.slice(chapter.startPosition, endPosition),
      };
    });
  }, [toc, content?.text]);

  // Handle scroll events - find which chapter span is visible
  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector(
      "[data-radix-scroll-area-viewport]",
    );
    if (!viewport || chapters.length === 0) return;

    const handleScroll = () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        const viewportRect = viewport.getBoundingClientRect();

        // Find the first chapter whose span is at or near the top of viewport
        let currentPosition = chapters[0].startPosition;
        for (const chapter of chapters) {
          const span = chapterRefs.current.get(chapter.id);
          if (span) {
            const rect = span.getBoundingClientRect();
            // If the chapter start is above the viewport bottom (visible)
            if (rect.top <= viewportRect.bottom) {
              currentPosition = chapter.startPosition;
            }
          }
        }

        onPositionChange?.(currentPosition);
      }, 100);
    };

    viewport.addEventListener("scroll", handleScroll);
    return () => {
      viewport.removeEventListener("scroll", handleScroll);
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [chapters, onPositionChange]);

  // Handle scroll to chapter
  useEffect(() => {
    if (scrollToChapter && scrollAreaRef.current) {
      const span = chapterRefs.current.get(scrollToChapter.id);
      if (span) {
        span.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  }, [scrollToChapter]);

  if (contentLoading) {
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
      <div className="max-w-4xl mx-auto px-8 py-12">
        <article className="prose prose-lg prose-slate max-w-none">
          <div className="whitespace-pre-wrap leading-relaxed text-foreground">
            {chapters.map((chapter) => (
              <span
                key={chapter.id}
                ref={(el) => {
                  if (el) chapterRefs.current.set(chapter.id, el);
                }}
              >
                {chapter.text}
              </span>
            ))}
          </div>
        </article>
      </div>
    </ScrollArea>
  );
};

export default ReadingView;
