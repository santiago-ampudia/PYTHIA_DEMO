"use client"

import { useEffect, useState } from "react"
import { ChatInterface, type Conversation } from "@/components/chat-interface"
import { getConversations } from "@/lib/actions/conversation"
import type { PaperRecommendationProps } from "@/components/paper-recommendation"

export default function SmartSearchPage() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadConversations = async () => {
      setIsLoading(true)
      const result = await getConversations("smart-search")
      if (result.success && result.data) {
        // Transform the data to match our Conversation interface
        const formattedConversations = result.data.map((conv: any) => ({
          id: conv.id,
          title: conv.title,
          messages: conv.messages.map((msg: any) => ({
            message: msg.content,
            isUser: msg.isUserMessage,
            timestamp: msg.timestamp,
            papers: msg.papers.map((paper: any) => ({
              id: paper.id,
              title: paper.title,
              authors: paper.authors,
              year: paper.year,
              journal: paper.journal,
              abstract: paper.abstract,
              url: paper.url || `https://arxiv.org/abs/${paper.arxivId}`,
              citations: paper.citations,
            })),
          })),
        }))
        setConversations(formattedConversations)
      }
      setIsLoading(false)
    }

    loadConversations()
  }, [])

  // Update the handleSubmit function to call the backend API
  const handleSubmit = async (message: string): Promise<{ text: string; papers: PaperRecommendationProps[] }> => {
    try {
      // Get the API URL from environment variables or use a default
      const apiUrl = process.env.ENGINE_API_URL || 'http://localhost:8000/research-query';
      console.log(`Sending query to API: ${apiUrl}`);
      
      // Show loading state in UI
      const loadingResponse = {
        text: "Processing your search query... This may take a minute or two.",
        papers: []
      };
      
      // Make the API call to the backend
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message, mode: 'smart-search' }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`API error: ${response.status} ${errorText}`);
        return {
          text: `Sorry, there was an error processing your search query. Please try again later. (Error: ${response.status})`,
          papers: [],
        };
      }
      
      // Parse the response
      const data = await response.json();
      console.log('API response:', data);
      
      // Debug the response structure
      console.log('Full API response structure:', JSON.stringify(data, null, 2));
      
      // Check if we need to manually read the answer file
      if (!data.text || data.text === 'No recommendations were generated. Please try a different query.') {
        console.warn('No recommendation text found in API response, checking for raw file paths:', data);
        
        // Try to extract recommendations from the response in any way possible
        let extractedText = '';
        
        // Option 1: Check if there's any string property that might contain our recommendations
        for (const key in data) {
          if (typeof data[key] === 'string' && data[key].length > 20) {
            console.log(`Found potential recommendations in field '${key}':`, data[key]);
            extractedText = data[key];
            break;
          }
        }
        
        if (extractedText) {
          return {
            text: extractedText,
            papers: data.papers || [],
          };
        }
      }
      
      return {
        text: data.text || 'No recommendations were generated. Please try a different query.',
        papers: data.papers || [],
      };
    } catch (error) {
      console.error('Error submitting query:', error);
      return {
        text: `Sorry, there was an error processing your search query: ${error instanceof Error ? error.message : 'Unknown error'}`,
        papers: [],
      };
    }
  }

  if (isLoading) {
    return <div className="flex h-screen items-center justify-center">Loading conversations...</div>
  }

  return (
    <ChatInterface
      title="Smart Search"
      description="Search for research papers with natural language queries"
      placeholder="Search for papers..."
      type="smart-search"
      onSubmit={handleSubmit}
      initialConversations={conversations}
      messageType="smart-search"
    />
  )
}
