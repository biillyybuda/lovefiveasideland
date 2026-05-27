-- Love Five subscription foundation.
--
-- Review before running in Supabase. This does not create Stripe webhooks or
-- payment handling; it gives each league a plan/status that the app can read.

ALTER TABLE public.leagues
ADD COLUMN IF NOT EXISTS plan_key text NOT NULL DEFAULT 'free',
ADD COLUMN IF NOT EXISTS subscription_status text NOT NULL DEFAULT 'active',
ADD COLUMN IF NOT EXISTS stripe_customer_id text,
ADD COLUMN IF NOT EXISTS stripe_subscription_id text,
ADD COLUMN IF NOT EXISTS trial_ends_at timestamptz,
ADD COLUMN IF NOT EXISTS current_period_ends_at timestamptz;

ALTER TABLE public.leagues
DROP CONSTRAINT IF EXISTS leagues_plan_key_check;

ALTER TABLE public.leagues
ADD CONSTRAINT leagues_plan_key_check
CHECK (plan_key IN ('free', 'pro', 'club'));

ALTER TABLE public.leagues
DROP CONSTRAINT IF EXISTS leagues_subscription_status_check;

ALTER TABLE public.leagues
ADD CONSTRAINT leagues_subscription_status_check
CHECK (subscription_status IN ('active', 'trialing', 'past_due', 'canceled', 'free'));

CREATE INDEX IF NOT EXISTS idx_leagues_plan_status
ON public.leagues (plan_key, subscription_status);

CREATE INDEX IF NOT EXISTS idx_leagues_stripe_customer
ON public.leagues (stripe_customer_id)
WHERE stripe_customer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_leagues_stripe_subscription
ON public.leagues (stripe_subscription_id)
WHERE stripe_subscription_id IS NOT NULL;

-- Optional later step:
-- Add RLS policies so only league owners/admins can update billing fields,
-- and only trusted server-side webhook code can write Stripe IDs/status.
