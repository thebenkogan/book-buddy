import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { getAISummary, askQuestion } from '@/services/bookService';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Send, Sparkles, BookMarked } from 'lucide-react';
import { ChatMessage } from '@/types/types';
import { motion, AnimatePresence } from 'framer-motion';

interface AIViewProps {
  bookId: string;
}

const AIView = ({ bookId }: AIViewProps) => {
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['aiSummary', bookId],
    queryFn: () => getAISummary(bookId),
  });

  const askMutation = useMutation({
    mutationFn: (q: string) => askQuestion(bookId, q),
    onSuccess: (answer, question) => {
      setChatHistory(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'user',
          content: question,
          timestamp: new Date(),
        },
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: answer,
          timestamp: new Date(),
        },
      ]);
      setQuestion('');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim()) {
      askMutation.mutate(question.trim());
    }
  };

  return (
    <ScrollArea className="h-full">
      <div className="max-w-4xl mx-auto px-8 py-8 space-y-6">
        {/* Summary Section */}
        <Card className="border-blue-200 bg-gradient-to-br from-blue-50/50 to-purple-50/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="text-blue-600" size={24} />
              AI Summary
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {summaryLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-4/5" />
              </div>
            ) : (
              <>
                <p className="text-foreground leading-relaxed">
                  {summary?.summary}
                </p>
                
                {summary?.keyPoints && summary.keyPoints.length > 0 && (
                  <div className="mt-4">
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      <BookMarked size={18} />
                      Key Points
                    </h4>
                    <ul className="space-y-2">
                      {summary.keyPoints.map((point, idx) => (
                        <li key={idx} className="flex gap-2">
                          <span className="text-blue-600 font-bold">•</span>
                          <span className="text-sm">{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Q&A Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="text-purple-600" size={24} />
              Ask Questions
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask anything about this book..."
                disabled={askMutation.isPending}
                className="flex-1"
              />
              <Button 
                type="submit" 
                disabled={!question.trim() || askMutation.isPending}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              >
                <Send size={18} />
              </Button>
            </form>

            {/* Chat History */}
            <div className="space-y-4 mt-6">
              <AnimatePresence>
                {chatHistory.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-4 ${
                        message.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-foreground border'
                      }`}
                    >
                      <p className="text-sm leading-relaxed">{message.content}</p>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {askMutation.isPending && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-start"
                >
                  <div className="bg-gray-100 rounded-lg p-4 border">
                    <div className="flex gap-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </motion.div>
              )}

              {chatHistory.length === 0 && !askMutation.isPending && (
                <div className="text-center text-muted-foreground text-sm py-8">
                  Ask a question to start the conversation
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
};

export default AIView;
