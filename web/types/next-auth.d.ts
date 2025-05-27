declare module "next-auth" {
  /**
   * Returned by `useSession`, `getSession` and received as a prop on the `SessionProvider` React Context
   */
  interface Session {
    user: {
      /** The user's id */
      id: string
      name?: string | null
      email?: string | null
      image?: string | null
      /** The authentication provider (github, google, credentials) */
      provider?: string
    }
  }
}
