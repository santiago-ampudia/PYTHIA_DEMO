"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, Plus, Trash2, Edit, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ChatMessage, type ChatMessageProps } from "@/components/chat-message"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { toast } from "@/components/ui/use-toast"
import type { PaperRecommendationProps } from "@/components/paper-recommendation"
import {
  createConversation,
  deleteConversation,
  addMessage,
  getConversations,
  renameConversation,
} from "@/lib/actions/conversation"

// Function to convert arXiv IDs in square brackets to hyperlinks
export const formatArXivLinks = (text: string): React.ReactNode[] => {
  if (!text) return []
  
  // Regular expression to match text within square brackets
  // This pattern specifically targets arXiv IDs which typically have formats like:
  // [1234.5678] or [arXiv:1234.5678] or [hep-ph/9876543]
  const regex = /\[(?:arXiv:)?(\d+\.\d+|[a-zA-Z-]+\/\d+)\]/g
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  let match
  
  // Keep track of unique arXiv IDs and their assigned numbers
  const arxivIdMap = new Map<string, number>()
  let sourceCounter = 1

  // First pass: identify all unique arXiv IDs and assign numbers
  let tempMatch
  while ((tempMatch = regex.exec(text)) !== null) {
    const arxivId = tempMatch[1]
    if (!arxivIdMap.has(arxivId)) {
      arxivIdMap.set(arxivId, sourceCounter++)
    }
  }
  
  // Reset regex index for second pass
  regex.lastIndex = 0

  // Second pass: build the parts array with numbered sources
  while ((match = regex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index))
    }
    
    // Add the link element for the match
    const arxivId = match[1]
    const sourceNumber = arxivIdMap.get(arxivId) || 0
    
    parts.push(
      <a 
        key={`arxiv-${match.index}`} 
        href={`https://arxiv.org/abs/${arxivId}`} 
        target="_blank" 
        rel="noopener noreferrer"
        className="text-blue-600 hover:underline font-medium"
        title={`arXiv ID: ${arxivId}`} // Add tooltip showing the actual ID
      >
        [{sourceNumber}]
      </a>
    )
    
    lastIndex = match.index + match[0].length
  }
  
  // Add any remaining text after the last match
  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex))
  }
  
  return parts
}

export interface Conversation {
  id: string
  title: string
  messages: (ChatMessageProps & { papers?: PaperRecommendationProps[] })[]
}

interface ChatInterfaceProps {
  title: string
  description: string
  placeholder: string
  type: "smart-answer" | "smart-search"
  onSubmit: (message: string) => Promise<{ text: string; papers: PaperRecommendationProps[] }>
  initialConversations?: Conversation[]
  messageType?: "smart-answer" | "smart-search"
}

export function ChatInterface({
  title,
  description,
  placeholder,
  type,
  onSubmit,
  initialConversations = [],
  messageType,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [conversations, setConversations] = useState<Conversation[]>(initialConversations)
  const [activeConversation, setActiveConversation] = useState<string | null>(
    initialConversations.length > 0 ? initialConversations[0].id : null,
  )
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Add these state variables inside the ChatInterface component
  const [isRenaming, setIsRenaming] = useState(false)
  const [newTitle, setNewTitle] = useState("")

  const currentConversation = conversations.find((conv) => conv.id === activeConversation) || {
    id: "new",
    title: "New Conversation",
    messages: [],
  }

  useEffect(() => {
    scrollToBottom()
  }, [currentConversation.messages])

  // Add a new useEffect to scroll to bottom when conversation changes
  useEffect(() => {
    scrollToBottom()
  }, [activeConversation])

  useEffect(() => {
    // Load conversations from the database
    const loadConversations = async () => {
      const result = await getConversations(type)
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
        if (formattedConversations.length > 0 && !activeConversation) {
          setActiveConversation(formattedConversations[0].id)
        }
      }
    }

    if (initialConversations.length === 0) {
      loadConversations()
    }
  }, [type, initialConversations.length, activeConversation])

  const scrollToBottom = () => {
    // Using setTimeout to ensure DOM is fully rendered before scrolling
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, 100)
  }

  // Add this function inside the ChatInterface component
  const handleRenameConversation = async () => {
    if (!activeConversation || activeConversation === "new" || !newTitle.trim()) return

    try {
      const result = await renameConversation(activeConversation, newTitle)

      if (result.success) {
        setConversations(
          conversations.map((conv) =>
            conv.id === activeConversation
              ? {
                  ...conv,
                  title: newTitle,
                }
              : conv,
          ),
        )
        toast({
          title: "Conversation renamed",
          description: "The conversation has been renamed successfully.",
        })
      } else {
        toast({
          title: "Error",
          description: result.error || "Failed to rename conversation",
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsRenaming(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    // Create user message with temporary ID
    const userMessage: ChatMessageProps = {
      message: input,
      isUser: true,
      timestamp: new Date().toISOString(),
      tempId: `temp-${Date.now()}`, // Add a temporary ID to track this message
    }

    // Store the current input before clearing it
    const currentInput = input
    setInput("")
    setIsLoading(true)

    // Create a local reference to the active conversation ID
    let conversationId = activeConversation
    let currentConversationMessages: (ChatMessageProps & { papers?: PaperRecommendationProps[] })[] = []

    // Optimistically update UI with user message immediately
    if (!activeConversation || activeConversation === "new") {
      // For new conversations, create a temporary conversation
      const tempConversation: Conversation = {
        id: `temp-${Date.now()}`,
        title: currentInput.length > 30 ? `${currentInput.substring(0, 30)}...` : currentInput,
        messages: [userMessage],
      }
      setConversations([tempConversation, ...conversations])
      setActiveConversation(tempConversation.id)
      currentConversationMessages = [userMessage]
    } else {
      // For existing conversations, add the message to the UI immediately
      const updatedConversations = conversations.map((conv) => {
        if (conv.id === activeConversation) {
          const updatedMessages = [...conv.messages, userMessage]
          currentConversationMessages = updatedMessages
          return {
            ...conv,
            messages: updatedMessages,
          }
        }
        return conv
      })
      setConversations(updatedConversations)
    }

    try {
      // Handle database operations and AI response
      if (!conversationId || conversationId === "new") {
        // Create new conversation in database
        const result = await createConversation(
          currentInput.length > 30 ? `${currentInput.substring(0, 30)}...` : currentInput, 
          type, 
          currentInput
        )

        if (result.success && result.data) {
          conversationId = result.data.id
          // Update the temporary conversation with the real ID
          setConversations(prevConversations => {
            return prevConversations.map((conv, index) => {
              if (index === 0 && conv.id.startsWith('temp-')) {
                return {
                  ...conv,
                  id: result.data.id,
                  title: result.data.title,
                }
              }
              return conv
            })
          })
          setActiveConversation(result.data.id)
        } else {
          toast({
            title: "Error",
            description: result.error || "Failed to create conversation",
            variant: "destructive",
          })
          // Even if there's an error, we keep the message visible in the UI
          return
        }
      } else {
        // Add user message to existing conversation in database
        await addMessage(conversationId, currentInput, true)
      }

      // Get AI response
      const response = await onSubmit(currentInput)

      // Create AI message
      const aiMessage: ChatMessageProps = {
        message: response.text,
        isUser: false,
        timestamp: new Date().toISOString(),
        papers: response.papers,
        formattedContent: response.text ? formatArXivLinks(response.text) : undefined
      }

      // Add AI response to database
      if (conversationId) {
        await addMessage(
          conversationId,
          response.text,
          false,
          response.papers.map((p) => p.id),
        )
      }

      // Update conversation state with AI response while preserving user message
      setConversations(prevConversations => {
        return prevConversations.map(conv => {
          if (conv.id === activeConversation || 
              (conversationId && conv.id === conversationId) ||
              (conv.id.startsWith('temp-') && prevConversations.indexOf(conv) === 0)) {
            return {
              ...conv,
              id: conversationId || conv.id, // Use real ID if available
              messages: [...currentConversationMessages, aiMessage],
            }
          }
          return conv
        })
      })
    } catch (error) {
      console.error('Error in chat submission:', error)
      toast({
        title: "Error",
        description: "Failed to get a response. Please try again.",
        variant: "destructive",
      })
      // Even on error, we keep the user message visible
    } finally {
      setIsLoading(false)
    }
  }

  const startNewConversation = () => {
    setActiveConversation("new")
  }

  const clearConversations = async () => {
    // Delete all conversations from the database
    for (const conversation of conversations) {
      await deleteConversation(conversation.id)
    }

    setConversations([])
    setActiveConversation("new")
    toast({
      title: "Conversations cleared",
      description: "All your conversations have been cleared.",
    })
  }

  const handleDeleteConversation = async () => {
    if (activeConversation && activeConversation !== "new") {
      await deleteConversation(activeConversation)

      setConversations(conversations.filter((conv) => conv.id !== activeConversation))
      setActiveConversation(conversations.length > 1 ? conversations[0].id : "new")
    }
  }

  return (
    <div className="flex h-[calc(100vh-2rem)] flex-col md:flex-row w-full">
      {/* Sidebar with conversation history */}
      <div className="w-full md:w-64 md:min-w-64 md:flex-shrink-0 border-r">
        <div className="flex h-14 items-center justify-between border-b px-4">
          <h2 className="font-semibold">{title}</h2>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <Plus className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={startNewConversation}>New Conversation</DropdownMenuItem>
              <DropdownMenuItem onClick={clearConversations} className="text-destructive">
                Clear All
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <ScrollArea className="h-[calc(100vh-3.5rem)]">
          <div className="flex flex-col gap-1 p-2">
            <Button
              variant={activeConversation === "new" ? "secondary" : "ghost"}
              className="justify-start"
              onClick={startNewConversation}
            >
              <Plus className="mr-2 h-4 w-4" />
              New Conversation
            </Button>
            {conversations.map((conversation) => (
              <Button
                key={conversation.id}
                variant={activeConversation === conversation.id ? "secondary" : "ghost"}
                className="justify-start overflow-hidden text-ellipsis whitespace-nowrap"
                onClick={() => setActiveConversation(conversation.id)}
              >
                {conversation.title}
              </Button>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Main chat area */}
      <div className="flex flex-1 flex-col w-full">
        <div className="flex h-14 items-center justify-between border-b px-4">
          <div className="flex items-center">
            {isRenaming && activeConversation && activeConversation !== "new" ? (
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  handleRenameConversation()
                }}
                className="flex items-center gap-2"
              >
                <Input
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="h-8 w-48"
                  autoFocus
                  placeholder="Enter new title..."
                />
                <Button type="submit" size="icon" variant="ghost" className="h-8 w-8">
                  <Check className="h-4 w-4" />
                </Button>
              </form>
            ) : (
              <h2 className="font-semibold">{currentConversation.title}</h2>
            )}
            {activeConversation && activeConversation !== "new" && !isRenaming && (
              <Button
                variant="ghost"
                size="icon"
                className="ml-2 h-8 w-8"
                onClick={() => {
                  setNewTitle(currentConversation.title)
                  setIsRenaming(true)
                }}
              >
                <Edit className="h-4 w-4" />
              </Button>
            )}
          </div>
          <div className="flex items-center gap-2">
            {activeConversation && activeConversation !== "new" && (
              <Button variant="ghost" size="icon" onClick={handleDeleteConversation}>
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <Plus className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={startNewConversation}>New Conversation</DropdownMenuItem>
                <DropdownMenuItem onClick={clearConversations} className="text-destructive">
                  Clear All
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        <ScrollArea className="flex-1 p-4 w-full">
          {currentConversation.messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center">
              <p className="text-center text-muted-foreground">No messages yet. Start a conversation!</p>
            </div>
          ) : (
            <div className="flex flex-col gap-4 w-full">
              {currentConversation.messages.map((msg, index) => (
                <ChatMessage key={index} {...msg} type={messageType || type} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>

        <Separator />

        <form onSubmit={handleSubmit} className="flex gap-2 p-4 w-full">
          <Input
            placeholder={placeholder}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1"
            disabled={isLoading}
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? "Thinking..." : <Send className="h-4 w-4" />}
          </Button>
        </form>
      </div>
    </div>
  )
}
