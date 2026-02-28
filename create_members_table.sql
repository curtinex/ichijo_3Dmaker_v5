--  テーブル作成
CREATE TABLE IF NOT EXISTS members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stripe_customer_id TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    payment_status TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_access_at TIMESTAMP WITH TIME ZONE
);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $ $
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$ $ language 'plpgsql';

DROP TRIGGER IF EXISTS update_members_updated_at ON members;

CREATE TRIGGER update_members_updated_at
    BEFORE UPDATE ON members
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

--  RLSの有効化
ALTER TABLE members ENABLE ROW LEVEL SECURITY;

-- 既存の古いポリシーがあれば削除
DROP POLICY IF EXISTS "Allow anonymous read" ON members;
DROP POLICY IF EXISTS "Allow service role to manage members" ON members;
DROP POLICY IF EXISTS "Users can view own data by email" ON members;

--  ChatGPTが提案していた「ログインしたユーザーのみが、自分自身のデータを閲覧できる」ポリシー
CREATE POLICY "Users can view own data by email"
ON members FOR SELECT
TO authenticated
USING (auth.jwt() ->> 'email' = email);

--  RenderのWebhookサーバー用（Stripeからの通知を書き込むため）のポリシー
CREATE POLICY "Allow service role to manage members"
ON members
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);
