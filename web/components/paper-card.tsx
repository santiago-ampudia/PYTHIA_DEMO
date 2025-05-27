"use client"

import { useState } from "react"
import { Bookmark, Share2, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { toast } from "@/components/ui/use-toast"
import { savePaper, removeSavedPaper } from "@/lib/actions/paper"

interface PaperCardProps {
  id: string
  title: string
  authors: string
  journal: string
  abstract: string
  citations: number
  year: number
  field: string
  isSaved?: boolean
  notes?: string
}

export function PaperCard({
  id,
  title,
  authors,
  journal,
  abstract,
  citations,
  year,
  field,
  isSaved: initialIsSaved = false,
  notes,
}: PaperCardProps) {
  const [isSaved, setIsSaved] = useState(initialIsSaved)
  const [isExpanded, setIsExpanded] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleSave = async () => {
    setIsLoading(true)
    try {
      if (isSaved) {
        const result = await removeSavedPaper(id)
        if (result.success) {
          setIsSaved(false)
          toast({
            title: "Paper removed from saved",
            description: "The paper has been removed from your saved papers.",
          })
        } else {
          toast({
            title: "Error",
            description: result.error || "Failed to remove paper",
            variant: "destructive",
          })
        }
      } else {
        const result = await savePaper(id, notes)
        if (result.success) {
          setIsSaved(true)
          toast({
            title: "Paper saved",
            description: "The paper has been added to your saved papers.",
          })
        } else {
          toast({
            title: "Error",
            description: result.error || "Failed to save paper",
            variant: "destructive",
          })
        }
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleShare = () => {
    // In a real app, this would copy a link to the clipboard or open a share dialog
    toast({
      title: "Share link copied",
      description: "A link to this paper has been copied to your clipboard.",
    })
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex justify-between">
          <div className="space-y-1">
            <CardTitle className="text-xl font-semibold">{title}</CardTitle>
            <p className="text-sm text-muted-foreground">{authors}</p>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span>{journal}</span>
              <span>•</span>
              <span>{year}</span>
              <span>•</span>
              <span>Citations: {citations.toLocaleString()}</span>
              <Badge variant="outline">{field}</Badge>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p className={`text-sm ${isExpanded ? "" : "line-clamp-3"}`}>{abstract}</p>
        {abstract.length > 200 && (
          <Button variant="link" className="p-0 h-auto text-xs mt-1" onClick={() => setIsExpanded(!isExpanded)}>
            {isExpanded ? "Show less" : "Show more"}
          </Button>
        )}
        {notes && (
          <div className="mt-2 p-2 bg-muted rounded-md">
            <p className="text-xs font-medium">Your notes:</p>
            <p className="text-sm">{notes}</p>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={handleSave} disabled={isLoading}>
            <Bookmark className={`h-4 w-4 mr-1 ${isSaved ? "fill-current" : ""}`} />
            {isLoading ? "Processing..." : isSaved ? "Saved" : "Save"}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleShare}>
            <Share2 className="h-4 w-4 mr-1" />
            Share
          </Button>
        </div>
        <Button variant="outline" size="sm" asChild>
          <a href={`https://arxiv.org/abs/${id}`} target="_blank" rel="noopener noreferrer">
            <ExternalLink className="h-4 w-4 mr-1" />
            View Paper
          </a>
        </Button>
      </CardFooter>
    </Card>
  )
}
