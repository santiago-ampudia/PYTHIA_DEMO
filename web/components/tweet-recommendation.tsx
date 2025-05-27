"use client"

import { ExternalLink, BookmarkPlus, BookmarkX } from "lucide-react"
import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { savePaper, removeSavedPaper } from "@/lib/actions/paper"
import { useState, useEffect } from "react"
import { toast } from "@/components/ui/use-toast"

export interface TweetRecommendationProps {
  content: string
  paperIds: string[]
  paperUrls?: Record<string, string>
  citationMap?: Map<string, number>
}

export function TweetRecommendation({
  content,
  paperIds,
  paperUrls = {},
  citationMap,
}: TweetRecommendationProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [isSaved, setIsSaved] = useState(false)
  const [formattedPaperIds, setFormattedPaperIds] = useState<string[]>([])

  // Function to format the tweet content with numeric citations using the global citation map
  const formatTweetContent = (text: string) => {
    // Regular expression to match text within square brackets
    // This pattern specifically targets arXiv IDs which typically have formats like:
    // [1234.5678] or [arXiv:1234.5678] or [hep-ph/9876543]
    const regex = /\[(?:arXiv:)?([a-zA-Z0-9\.-]+\/?\d*)\]/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    
    // Use the provided global citation map or create a local one if not provided
    const useGlobalMap = citationMap && citationMap.size > 0;
    const localCitationMap = new Map<string, number>();
    
    if (!useGlobalMap) {
      // Create a local map if global one isn't provided
      let sourceCounter = 1;
      // First pass: identify all unique arXiv IDs and assign numbers
      let tempMatch;
      const tempText = text;
      while ((tempMatch = regex.exec(tempText)) !== null) {
        const arxivId = tempMatch[1];
        if (!localCitationMap.has(arxivId)) {
          localCitationMap.set(arxivId, sourceCounter++);
        }
      }
      // Reset regex index
      regex.lastIndex = 0;
    }
    
    // Process the text and replace citations
    let match;
    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      
      // Extract the arXiv ID
      const arxivId = match[1];
      
      // Get citation number from global map if available, otherwise use local map
      const sourceNumber = useGlobalMap 
        ? citationMap!.get(arxivId) || 0
        : localCitationMap.get(arxivId) || 0;
      
      // Add the citation number as a link
      parts.push(
        <a
          key={`arxiv-${match.index}`}
          href={`https://arxiv.org/abs/${arxivId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline font-medium"
          title={`arXiv ID: ${arxivId}`}
        >
          [{sourceNumber}]
        </a>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add any remaining text after the last match
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts;
  };

  // Format paper IDs when component mounts
  useEffect(() => {
    // Format each paper ID to ensure it's in the correct format for the database
    const formatted = paperIds.map(id => {
      // Remove 'arXiv:' prefix if present
      if (id.startsWith('arXiv:')) {
        return id.substring(6);
      }
      return id;
    });
    setFormattedPaperIds(formatted);
  }, [paperIds]);

  // We don't automatically check if papers are saved - we'll only save when the user clicks the button

  const handleSaveAllPapers = async () => {
    if (isSaved) {
      // If already saved, unsave all papers
      await handleUnsaveAllPapers();
      return;
    }

    setIsLoading(true);
    try {
      // Log the paper IDs being saved for debugging
      console.log('Saving papers with IDs:', formattedPaperIds);
      
      // Save each paper ID
      const savePromises = formattedPaperIds.map(id => savePaper(id));
      const results = await Promise.all(savePromises);
      
      // Check results - if any were successful or already saved, consider it a success
      const successCount = results.filter(result => 
        result.success || result.error === "Paper already saved"
      ).length;
      
      if (successCount > 0) {
        // Set saved state to true regardless of partial success
        setIsSaved(true);
        
        toast({
          title: "Papers saved",
          description: "Papers from this recommendation have been saved to your collection.",
        });
      } else {
        // Check for specific errors
        const errors = results
          .filter(result => result.error)
          .map(result => result.error);
        
        if (errors.includes("Paper not found")) {
          toast({
            title: "Papers not found",
            description: "Some papers could not be saved because they were not found in the database. This may happen with placeholder or test paper IDs.",
            variant: "destructive",
          });
        } else {
          toast({
            title: "Error saving papers",
            description: `None of the papers could be saved. Please try again later.`,
            variant: "destructive",
          });
        }
      }
    } catch (error) {
      console.error('Error saving papers:', error);
      toast({
        title: "Error",
        description: "An unexpected error occurred while saving papers.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleUnsaveAllPapers = async () => {
    setIsLoading(true);
    try {
      // Unsave each paper ID
      const removePromises = formattedPaperIds.map(id => removeSavedPaper(id));
      const results = await Promise.all(removePromises);
      
      // Check if all papers were unsaved successfully
      const allSuccessful = results.every(result => result.success);
      
      if (allSuccessful) {
        setIsSaved(false);
        toast({
          title: "Papers removed",
          description: "All papers have been removed from your collection.",
        });
      } else {
        // Count successful removals
        const successCount = results.filter(result => result.success).length;
        
        // If all papers were removed, consider the set as unsaved
        if (successCount === formattedPaperIds.length) {
          setIsSaved(false);
        }
        
        toast({
          title: "Partially removed",
          description: `${successCount} out of ${formattedPaperIds.length} papers were removed successfully.`,
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error removing papers:', error);
      toast({
        title: "Error",
        description: "An unexpected error occurred while removing papers.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="overflow-hidden border-l-4 border-l-primary w-full">
      <CardContent className="p-4 overflow-x-auto">
        <div className="text-sm whitespace-pre-wrap">
          {formatTweetContent(content)}
        </div>
      </CardContent>
      <CardFooter className="p-3 pt-0 flex flex-wrap justify-between items-center">
        <div className="flex flex-wrap items-center text-sm text-muted-foreground">
          {paperIds.length > 0 ? (
            // Filter out duplicate paper IDs and display each one only once
            [...new Set(paperIds)].map((id, index, uniqueIds) => {
              // Use the global citation map if provided, otherwise create a local one
              const useGlobalMap = citationMap && citationMap.size > 0;
              let sourceNumber = 0;
              
              if (useGlobalMap) {
                sourceNumber = citationMap!.get(id) || 0;
              } else {
                // Create a local map if global one isn't provided
                const localMap = new Map<string, number>();
                let counter = 1;
                paperIds.forEach(paperId => {
                  if (!localMap.has(paperId)) {
                    localMap.set(paperId, counter++);
                  }
                });
                sourceNumber = localMap.get(id) || 0;
              }
              const arxivUrl = id.startsWith('arXiv:') 
                ? `https://arxiv.org/abs/${id.substring(6)}` 
                : `https://arxiv.org/abs/${id}`;
              
              return (
                <a
                  key={`${id}-${index}`}
                  href={paperUrls[id] || arxivUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline flex items-center mr-3 mb-1"
                >
                  <ExternalLink className="h-3 w-3 mr-1" />
                  <span>[{sourceNumber}]</span>
                  <span className="ml-1 font-medium">arXiv: {id.replace(/^arXiv:/, '')}</span>
                  {index < uniqueIds.length - 1 && ", "}
                </a>
              );
            })
          ) : (
            <span className="text-muted-foreground">No papers found.</span>
          )}
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleSaveAllPapers}
          disabled={isLoading}
          className="flex items-center gap-1"
        >
          {isLoading ? (
            <>
              <span className="mr-2">{isSaved ? 'Removing...' : 'Saving...'}</span>
            </>
          ) : isSaved ? (
            <>
              <BookmarkX className="h-4 w-4 mr-2" />
              <span>Unsave All</span>
            </>
          ) : (
            <>
              <BookmarkPlus className="h-4 w-4 mr-2" />
              <span>Save All</span>
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}
