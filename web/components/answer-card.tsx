"use client"

import { useState } from "react"
import { ThumbsUp, Bookmark, Share2, MessageSquare } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { toast } from "@/components/ui/use-toast"

interface Paper {
  title: string
  authors: string
  year: number
  url: string
}

interface AnswerCardProps {
  question: string
  answer: string
  papers: Paper[]
  upvotes: number
  timestamp: string
}

export function AnswerCard({ question, answer, papers, upvotes: initialUpvotes, timestamp }: AnswerCardProps) {
  const [isUpvoted, setIsUpvoted] = useState(false)
  const [upvotes, setUpvotes] = useState(initialUpvotes)
  const [isSaved, setIsSaved] = useState(false)

  const handleUpvote = () => {
    if (isUpvoted) {
      setUpvotes(upvotes - 1)
    } else {
      setUpvotes(upvotes + 1)
    }
    setIsUpvoted(!isUpvoted)
  }

  const handleSave = () => {
    setIsSaved(!isSaved)
    toast({
      title: isSaved ? "Answer removed from saved" : "Answer saved",
      description: isSaved
        ? "The answer has been removed from your saved answers."
        : "The answer has been added to your saved answers.",
    })
  }

  const handleShare = () => {
    // In a real app, this would copy a link to the clipboard or open a share dialog
    toast({
      title: "Share link copied",
      description: "A link to this answer has been copied to your clipboard.",
    })
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-semibold">{question}</CardTitle>
        <p className="text-sm text-muted-foreground">{timestamp}</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="whitespace-pre-line text-sm">{answer}</div>

        <Collapsible className="w-full">
          <CollapsibleTrigger asChild>
            <Button variant="outline" size="sm" className="w-full">
              View {papers.length} cited papers
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4 space-y-2">
            {papers.map((paper, index) => (
              <div key={index} className="border rounded-md p-3 text-sm">
                <p className="font-medium">{paper.title}</p>
                <p className="text-muted-foreground">
                  {paper.authors} ({paper.year})
                </p>
                <Button variant="link" size="sm" className="p-0 h-auto mt-1" asChild>
                  <a href={paper.url} target="_blank" rel="noopener noreferrer">
                    View Paper
                  </a>
                </Button>
              </div>
            ))}
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
      <CardFooter className="flex justify-between">
        <div className="flex gap-2">
          <Button variant={isUpvoted ? "default" : "ghost"} size="sm" onClick={handleUpvote} className="gap-1">
            <ThumbsUp className="h-4 w-4" />
            {upvotes > 0 && upvotes}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleSave}>
            <Bookmark className={`h-4 w-4 mr-1 ${isSaved ? "fill-current" : ""}`} />
            {isSaved ? "Saved" : "Save"}
          </Button>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={handleShare}>
            <Share2 className="h-4 w-4 mr-1" />
            Share
          </Button>
          <Button variant="ghost" size="sm" asChild>
            <a href="#">
              <MessageSquare className="h-4 w-4 mr-1" />
              Follow-up
            </a>
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}
