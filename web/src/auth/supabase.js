import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://zvaidwubfhjlzorxrkbg.supabase.co';
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2YWlkd3ViZmhqbHpvcnhya2JnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODkxNDg3NDQsImV4cCI6MjEwNDcyNDc0NH0.U8_707zT-LXgdQTBkXbnVOTdWy1HEBtRjrfqDa5uizQ';

export const supabase = createClient(supabaseUrl, supabaseKey);
