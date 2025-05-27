import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { PaperRecommendation, type PaperRecommendationProps } from "@/components/paper-recommendation"
import { TweetRecommendation } from "@/components/tweet-recommendation"
import { useEffect, useState } from "react";
import type React from "react"

export interface ChatMessageProps {
  message: string
  isUser: boolean
  timestamp: string
  avatar?: string
  papers?: PaperRecommendationProps[]
  formattedContent?: React.ReactNode[]
  type?: "smart-answer" | "smart-search"
  tempId?: string // Temporary ID for tracking messages before database confirmation
}

export function ChatMessage({ message, isUser, timestamp, avatar, papers = [], formattedContent, type = "smart-answer" }: ChatMessageProps) {
  const [displayTime, setDisplayTime] = useState("");
  useEffect(() => {
    if (timestamp) {
      // Check if timestamp is already a formatted string or a date that needs formatting
      try {
        // Try to parse it as a date first
        const date = new Date(timestamp);
        // If it's a valid date (not Invalid Date), format it
        if (!isNaN(date.getTime())) {
          setDisplayTime(date.toLocaleTimeString());
        } else {
          // If it's already a formatted string, use it directly
          setDisplayTime(timestamp);
        }
      } catch (error) {
        // If there's an error, just use the timestamp as is
        setDisplayTime(timestamp);
      }
    }
  }, [timestamp]);

  // Parse the message to extract individual tweet recommendations if in smart-search mode
  const [tweetRecommendations, setTweetRecommendations] = useState<{ content: string; paperIds: string[]; citationMap: Map<string, number> }[]>([]);
  
  useEffect(() => {
    if (type === "smart-search" && !isUser && message) {
      try {
        // Try to parse the message as individual recommendations
        // Format is expected to be "Recommendation X: [content]" separated by blank lines
        const recommendations: { content: string; paperIds: string[]; citationMap: Map<string, number> }[] = [];
        const lines = message.split('\n\n');
        
        // First, collect all unique paper IDs across all recommendations
        const allPaperIds = new Set<string>();
        const citationNumberMap = new Map<string, number>();
        
        // First pass: collect all paper IDs
        lines.forEach(line => {
          if (line.trim().startsWith('Recommendation')) {
            const paperIdRegex = /\[((?:arXiv:)?[a-zA-Z0-9\.-]+\/?\d*)\)/g;
            const matches = [...line.matchAll(paperIdRegex)];
            matches.forEach(match => {
              const id = match[1];
              if (!/^\d+$/.test(id)) { // Skip pure numeric IDs as they're likely citation numbers
                allPaperIds.add(id);
              }
            });
            
            // Also check for numeric citations and extract them
            const numericCitationRegex = /\[(\d+)\]/g;
            const numMatches = [...line.matchAll(numericCitationRegex)];
            numMatches.forEach(match => {
              const num = match[1];
              // For testing, create a placeholder ID
              allPaperIds.add(`2304.${num.padStart(5, '0')}`);
            });
          }
        });
        
        // Assign global citation numbers to all unique paper IDs
        let citationCounter = 1;
        Array.from(allPaperIds).forEach(id => {
          citationNumberMap.set(id, citationCounter++);
        });
        
        // Second pass: process each recommendation with the global citation map
        lines.forEach(line => {
          if (line.trim().startsWith('Recommendation')) {
            // Extract paper IDs from the content using regex
            // This regex matches arXiv IDs in the format [1234.5678] or [arXiv:1234.5678] or [hep-ph/9876543]
            const paperIdRegex = /\[((?:arXiv:)?[a-zA-Z0-9\.-]+\/?\d*)\]/g;
            const matches = [...line.matchAll(paperIdRegex)];
            // Make sure we have the full arXiv ID format
            // Use a Set to ensure we only have unique IDs
            const paperIdsSet = new Set<string>();
            
            matches.forEach(match => {
              const id = match[1];
              // If it's just a number (like [1]), it's likely a citation number, not an actual arXiv ID
              // In this case, use a placeholder ID or skip it
              if (!/^\d+$/.test(id)) {
                paperIdsSet.add(id);
              } else {
                // For testing purposes, use a placeholder arXiv ID
                paperIdsSet.add(`2304.${id.padStart(5, '0')}`);
              }
            });
            
            const paperIds = Array.from(paperIdsSet);
            
            // Remove the "Recommendation X:" prefix from the content
            let cleanContent = line.trim();
            if (cleanContent.startsWith('Recommendation')) {
              // Find the first colon and remove everything before it and the colon itself
              const colonIndex = cleanContent.indexOf(':');
              if (colonIndex !== -1) {
                cleanContent = cleanContent.substring(colonIndex + 1).trim();
              }
            }
            
            recommendations.push({
              content: cleanContent,
              paperIds: paperIds,
              citationMap: citationNumberMap
            });
          }
        });
        
        setTweetRecommendations(recommendations);
      } catch (error) {
        console.error("Error parsing tweet recommendations:", error);
        setTweetRecommendations([]);
      }
    }
  }, [message, type, isUser]);

  return (
    <div className={cn("flex w-full gap-3 p-4", isUser ? "justify-end" : "")}>
      {!isUser && (
        <Avatar className="h-8 w-8 mt-1">
          <AvatarImage src={avatar || "/placeholder.svg?height=32&width=32&query=AI"} alt="AI" />
          <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      )}
      <div className={cn("flex flex-col gap-2", isUser ? "items-end max-w-[85%]" : "w-full")}>
        {isUser ? (
          <Card className="bg-primary text-primary-foreground">
            <CardContent className="p-3">
              <div className="whitespace-pre-line text-sm">{message}</div>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {/* For smart-search mode, show individual tweet recommendations */}
            {type === "smart-search" && tweetRecommendations.length > 0 ? (
              <div className="space-y-4">
                {tweetRecommendations.map((tweet, index) => (
                  <TweetRecommendation 
                    key={index} 
                    content={tweet.content} 
                    paperIds={tweet.paperIds}
                    citationMap={tweet.citationMap}
                  />
                ))}
              </div>
            ) : (
              <>
                {/* For smart-answer mode or if no tweet recommendations were parsed, show the regular message card */}
                {(type === "smart-answer" || tweetRecommendations.length === 0) && (
                  <Card>
                    <CardContent className="p-3">
                      <div className="text-sm">
                        {formattedContent && formattedContent.length > 0 ? (
                          <div className="whitespace-pre-wrap">
                            {formattedContent}
                          </div>
                        ) : (
                          <div className="whitespace-pre-line">
                            {message}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Paper recommendations box removed as requested */}
                {/* Keeping this commented out in case it needs to be restored later
                {type === "smart-answer" && papers.length > 0 && (
                  <div className="space-y-3">
                    {papers.map((paper, index) => (
                      <PaperRecommendation key={index} {...paper} />
                    ))}
                  </div>
                )}
                */}
              </>
            )}
          </div>
        )}
        <span className="text-xs text-muted-foreground">{displayTime}</span>
      </div>
      {isUser && (
        <Avatar className="h-8 w-8 mt-1">
          <AvatarImage src={avatar || "/abstract-geometric-shapes.png"} alt="User" />
          <AvatarFallback>US</AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}
