import type { NextAuthOptions } from "next-auth"
import NextAuth from "next-auth/next"
import CredentialsProvider from "next-auth/providers/credentials"
import GithubProvider from "next-auth/providers/github"
import GoogleProvider from "next-auth/providers/google"
import { PrismaAdapter } from "@auth/prisma-adapter"
import { compare } from "bcrypt"
import prisma from "@/lib/db"

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  providers: [
    GithubProvider({
      clientId: process.env.GITHUB_CLIENT_ID as string,
      clientSecret: process.env.GITHUB_CLIENT_SECRET as string,
      profile(profile) {
        return {
          id: profile.id.toString(),
          name: profile.name || profile.login,
          email: profile.email,
          image: profile.avatar_url,
        }
      },
      authorization: {
        params: {
          scope: "read:user user:email repo",
        },
      },
      httpOptions: {
        timeout: 10000, // Increase timeout to 10 seconds (from default 3.5s)
      },
    }),
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID as string,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
      authorization: {
        params: {
          prompt: "consent",
          access_type: "offline",
          response_type: "code",
        },
      },
    }),
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        const user = await prisma.user.findUnique({
          where: {
            email: credentials.email,
          },
        })

        if (!user || !user.password) {
          return null
        }

        const isPasswordValid = await compare(credentials.password, user.password)

        if (!isPasswordValid) {
          return null
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image,
        }
      },
    }),
  ],
  session: {
    strategy: "jwt",
  },
  pages: {
    signIn: "/login",
  },
  callbacks: {
    async session({ session, token }) {
      if (token) {
        session.user.id = token.id as string
        // Add provider information to the session
        session.user.provider = token.provider as string
      }
      return session
    },
    async jwt({ token, user, account }) {
      if (user) {
        token.id = user.id
      }

      // Store the provider in the token
      if (account) {
        token.provider = account.provider
      }

      // Store GitHub access token in the token
      if (account && account.provider === "github" && account.access_token) {
        token.githubAccessToken = account.access_token

        // Save GitHub profile in the database
        try {
          // Check if user already has a GitHub profile
          const existingProfile = await prisma.githubProfile.findUnique({
            where: {
              userId: token.id as string,
            },
          })

          // Get GitHub username
          const response = await fetch("https://api.github.com/user", {
            headers: {
              Authorization: `token ${account.access_token}`,
              Accept: "application/vnd.github.v3+json",
            },
          })

          if (response.ok) {
            const githubUser = await response.json()

            if (existingProfile) {
              // Update existing profile
              await prisma.githubProfile.update({
                where: {
                  id: existingProfile.id,
                },
                data: {
                  githubUsername: githubUser.login,
                  accessToken: account.access_token,
                  refreshToken: account.refresh_token,
                  expiresAt: account.expires_at ? new Date(account.expires_at * 1000) : null,
                },
              })
            } else {
              // Create new profile
              await prisma.githubProfile.create({
                data: {
                  userId: token.id as string,
                  githubUsername: githubUser.login,
                  accessToken: account.access_token,
                  refreshToken: account.refresh_token,
                  expiresAt: account.expires_at ? new Date(account.expires_at * 1000) : null,
                },
              })
            }
          }
        } catch (error) {
          console.error("Error saving GitHub profile:", error)
        }
      }

      return token
    },
  },
}

const handler = NextAuth(authOptions)

export { handler as GET, handler as POST }
