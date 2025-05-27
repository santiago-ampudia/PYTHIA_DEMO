import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TweetRecommendation } from "@/components/tweet-recommendation";

interface Tweet {
  text: string;
  paper_ids: string[];
}

interface Category {
  name: string;
  description: string;
  tweets: Tweet[];
}

interface CategorizedTweetRecommendationsProps {
  categories: Category[];
}

export function CategorizedTweetRecommendations({ categories }: CategorizedTweetRecommendationsProps) {
  return (
    <div className="space-y-8">
      {categories.map((category, index) => (
        <div key={`category-${index}`} className="space-y-4">
          {/* Category Header */}
          <Card className="bg-muted/50 border-l-4 border-l-primary">
            <CardHeader className="py-3">
              <CardTitle className="text-lg">{category.name}</CardTitle>
              <p className="text-sm text-muted-foreground">{category.description}</p>
            </CardHeader>
          </Card>
          
          {/* Category Tweets */}
          <div className="space-y-4 pl-4">
            {category.tweets.map((tweet, tweetIndex) => (
              <TweetRecommendation
                key={`tweet-${index}-${tweetIndex}`}
                content={tweet.text}
                paperIds={tweet.paper_ids}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
