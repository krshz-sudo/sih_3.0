import { supabase } from './supabase.js';

export async function checkSession() {
  const { data: { session }, error } = await supabase.auth.getSession();
  return session;
}

export function handleLogout(redirectUrl = '/index.html') {
  return async () => {
    await supabase.auth.signOut();
    window.location.href = redirectUrl;
  };
}

// Global Auth State Observer
supabase.auth.onAuthStateChange((event, session) => {
  console.log('Auth event:', event);
  // Optional: Add global reaction to auth changes here if needed
});
