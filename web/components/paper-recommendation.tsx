"use client"

import { ExternalLink } from "lucide-react"
import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { savePaper, removeSavedPaper } from "@/lib/actions/paper"
import { useState } from "react"
import { toast } from "@/components/ui/use-toast"

export interface PaperRecommendationProps {
  id: string
  title: string
  authors: string
  year: number
  journal?: string
  abstract?: string
  url: string
  citations?: number
  isSaved?: boolean
}

export function PaperRecommendation({
  id,
  title,
  authors,
  year,
  journal,
  abstract,
  url,
  citations,
  isSaved: initialIsSaved = false,
}: PaperRecommendationProps) {
  const [isSaved, setIsSaved] = useState(initialIsSaved)
  const [isLoading, setIsLoading] = useState(false)

  const handleSaveToggle = async () => {
    setIsLoading(true)
    try {
      if (isSaved) {
        const result = await removeSavedPaper(id)
        if (result.success) {
          setIsSaved(false)
          toast({
            title: "Paper removed",
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
        const result = await savePaper(id)
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

  return (
    <Card className="overflow-hidden border-l-4 border-l-primary">
      <CardContent className="p-4 space-y-2">
        <div className="space-y-1">
          <h3 className="font-semibold text-base">{title}</h3>
          <p className="text-sm text-muted-foreground">{authors}</p>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {journal && <span className="font-medium">{journal}</span>}
            <Badge variant="outline" className="text-xs font-normal">
              {year}
            </Badge>
            {citations !== undefined && (
              <Badge variant="secondary" className="text-xs font-normal">
                {citations} citations
              </Badge>
            )}
          </div>
        </div>
        {abstract && <p className="text-sm line-clamp-3">{abstract}</p>}
      </CardContent>
      <CardFooter className="p-3 pt-0 flex justify-between">
        <Button variant="link" size="sm" className="px-0 h-auto text-primary" asChild>
          <a href={url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
            <ExternalLink className="h-3 w-3" />
            <span>arxiv.org</span>
          </a>
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleSaveToggle}
          disabled={isLoading}
          className={isSaved ? "text-primary" : ""}
        >
          {isLoading ? "Processing..." : isSaved ? "Saved" : "Save"}
        </Button>
      </CardFooter>
    </Card>
  )
}
