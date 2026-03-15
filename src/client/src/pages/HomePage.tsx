import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Search, BookOpen, Plus, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { toast } from "sonner";
import { SidebarTrigger } from "@/components/ui/sidebar";
import BookSidebar from "@/components/BookSidebar";
import {
  searchBooks,
  addBookToShelf,
  requestBook,
} from "@/services/bookService";
import { SearchBook } from "@/types/types";

const USER_ID = "default_user";

const HomePage = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [inputSearchQuery, setInputSearchQuery] = useState("");
  const queryClient = useQueryClient();

  const { data: searchData, isLoading } = useQuery({
    queryKey: ["search", searchQuery],
    queryFn: () => searchBooks(searchQuery || undefined),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
  });

  const addBookMutation = useMutation({
    mutationFn: (bookId: string) => addBookToShelf(bookId, USER_ID),
    onSuccess: () => {
      toast.success("Book added to your bookshelf!");
      queryClient.invalidateQueries({ queryKey: ["books"] });
    },
    onError: () => {
      toast.error("Failed to add book to shelf");
    },
  });

  const requestBookMutation = useMutation({
    mutationFn: (book: SearchBook) =>
      requestBook({
        userId: USER_ID,
        bookId: book.book_id,
        title: book.title,
        author: book.author,
      }),
    onSuccess: () => {
      toast.success("Book requested! We'll notify you when it's available.");
    },
    onError: () => {
      toast.error("Failed to request book");
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    setSearchQuery(inputSearchQuery);
    e.preventDefault();
  };

  const handleAddBook = (bookId: string) => {
    addBookMutation.mutate(bookId);
  };

  const handleRequestBook = (book: SearchBook) => {
    requestBookMutation.mutate(book);
  };

  return (
    <div className="flex min-h-screen w-full">
      <BookSidebar />

      <div className="flex flex-col flex-1 min-w-0">
        <header className="flex items-center gap-4 border-b bg-white px-6 py-4 sticky top-0 z-10">
          <SidebarTrigger />
          <div className="flex-1">
            <form onSubmit={handleSearch} className="flex gap-2 max-w-2xl">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <Input
                  type="text"
                  placeholder="Search for books..."
                  value={inputSearchQuery}
                  onChange={(e) => {
                    const query = e.target.value;
                    if (query.length == 0) {
                      setSearchQuery("");
                    }
                    setInputSearchQuery(query);
                  }}
                  className="pl-10"
                />
              </div>
              <Button type="submit" disabled={isLoading}>
                {isLoading ? <Loader2 className="animate-spin" /> : "Search"}
              </Button>
            </form>
          </div>
        </header>

        <main className="flex-1 max-w-4xl mx-auto px-6 py-6 w-full">
          {isLoading ? (
            <div className="flex justify-center py-12">
              <Loader2 className="animate-spin text-gray-400" size={32} />
            </div>
          ) : searchData?.results.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <BookOpen className="mx-auto mb-4" size={48} />
              <p>No books found. Try a different search term.</p>
            </div>
          ) : (
            <div className="grid gap-4">
              {searchData?.results.map((book) => {
                const bookId = book.title.replace(/ /g, "_").toLowerCase();
                return (
                  <Card key={book.book_id} className="overflow-hidden">
                    <CardContent className="p-4">
                      <div className="flex gap-4">
                        <div className="w-20 h-28 bg-gray-200 rounded flex-shrink-0 overflow-hidden">
                          {book.cover_url ? (
                            <img
                              src={book.cover_url}
                              alt={book.title}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <BookOpen className="text-gray-400" />
                            </div>
                          )}
                        </div>

                        <div className="flex-1 min-w-0">
                          <h3 className="font-semibold text-lg truncate">
                            {book.title}
                          </h3>
                          <p className="text-gray-600 text-sm">{book.author}</p>
                          {book.indexed && (
                            <span className="inline-block mt-1 text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">
                              In Library
                            </span>
                          )}
                        </div>

                        <div className="flex-shrink-0 flex items-start">
                          {book.indexed ? (
                            <Button
                              size="sm"
                              onClick={() => handleAddBook(bookId)}
                              disabled={addBookMutation.isPending}
                            >
                              {addBookMutation.isPending ? (
                                <Loader2 className="animate-spin" />
                              ) : (
                                <>
                                  <Plus className="mr-1" size={16} />
                                  Add
                                </>
                              )}
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleRequestBook(book)}
                              disabled={requestBookMutation.isPending}
                            >
                              {requestBookMutation.isPending ? (
                                <Loader2 className="animate-spin" />
                              ) : (
                                "Request"
                              )}
                            </Button>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default HomePage;
