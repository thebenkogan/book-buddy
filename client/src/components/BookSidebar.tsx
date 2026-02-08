import { useQuery } from '@tanstack/react-query';
import { useNavigate, useParams } from 'react-router-dom';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";
import { BookOpen, Library } from 'lucide-react';
import { getBooks } from '@/services/bookService';
import { Progress } from '@/components/ui/progress';

const BookSidebar = () => {
  const navigate = useNavigate();
  const { bookId } = useParams();
  
  const { data: books, isLoading } = useQuery({
    queryKey: ['books'],
    queryFn: getBooks,
  });

  return (
    <Sidebar className="border-r">
      <SidebarHeader className="border-b p-4">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-2 rounded-lg">
            <Library className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold">BookBuddy AI</h1>
            <p className="text-xs text-muted-foreground">Your Reading Library</p>
          </div>
        </div>
      </SidebarHeader>
      
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-xs uppercase tracking-wider">
            My Books ({books?.length || 0})
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {isLoading ? (
                <div className="p-4 text-sm text-muted-foreground">Loading books...</div>
              ) : books?.length === 0 ? (
                <div className="p-4 text-sm text-muted-foreground">No books yet</div>
              ) : (
                books?.map((book) => (
                  <SidebarMenuItem key={book.id}>
                    <SidebarMenuButton 
                      asChild 
                      isActive={bookId === book.id}
                      className="h-auto py-3"
                    >
                      <button onClick={() => navigate(`/book/${book.id}`)}>
                        <div className="flex items-start gap-3 w-full">
                          <BookOpen className="h-5 w-5 mt-0.5 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm truncate">{book.title}</div>
                            {book.author && (
                              <div className="text-xs text-muted-foreground truncate">
                                {book.author}
                              </div>
                            )}
                            <div className="mt-2 space-y-1">
                              <Progress value={book.progress} className="h-1" />
                              <div className="text-xs text-muted-foreground">
                                {book.progress}% complete
                              </div>
                            </div>
                          </div>
                        </div>
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
};

export default BookSidebar;
