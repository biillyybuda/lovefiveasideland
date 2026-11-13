# Password recovery

The login page offers **Forgot password?** and `/login?mode=reset`. It asks for the registered email address and uses Supabase's recovery-email API. It never deletes an account or creates a replacement account.

The email opens `/reset-password`, where the existing browser Supabase client validates the recovery session. The user enters and confirms a new password of at least eight characters. Supabase's configured password policy also applies. Successful updates clear the form and request sign-out before returning the user to sign in. Invalid, expired and reused links offer another reset email. A tab-local marker allows refreshing a verified recovery page; the Auth server still validates the user.

## Deployment setup

Deploy the `lovefive-web` changes before sending recovery emails. No database migration is needed for this feature.

In **Supabase → Authentication → URL Configuration**, set Site URL to the actual production origin and allow the exact reset page for every origin you serve:

- `https://lovefive.co.uk/reset-password`
- `https://www.lovefive.co.uk/reset-password` if that host serves the app
- `https://app.lovefive.co.uk/reset-password` if that host serves the app
- `http://localhost:3000/reset-password` for local testing only

The code uses the current browser origin, so the corresponding URL must be allowed. Avoid wildcard production redirects. In **Authentication → Email Templates → Reset Password**, the standard link should use `{{ .ConfirmationURL }}`. This app uses Supabase's default browser implicit flow, not an SSR/PKCE callback; no template token hashing or server callback is required.

Ensure production email delivery is configured and tested with an account you control. Supabase's default mail service can restrict delivery to project-team email addresses; use a configured SMTP provider to send resets to ordinary league members. Do not put SMTP credentials, service-role keys, recovery tokens or passwords into the repository.

After deployment, test an email from Forgot password, opening the link, refreshing the reset page, changing the password, and signing in with the new password. Also test an expired/reused link. Automated tests mock email delivery and do not send real recovery emails.

## Account records

`auth.users` is the login account. `public.profiles` is its app profile, and `public.league_members` controls league access. Deleting a profile does not reset its user's password or remove the login account. Restore the missing profile/membership against the original user ID and use password recovery. Player and match records should not be deleted to repair login access.
