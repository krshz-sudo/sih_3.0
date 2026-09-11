import { supabase } from './supabase.js';

document.addEventListener('DOMContentLoaded', () => {
  const loginForm = document.getElementById('login-form');
  const btnGoogle = document.getElementById('btn-google');
  const btnSubmit = document.getElementById('btn-submit');
  const alertBox = document.getElementById('alert-box');
  const togglePwd = document.getElementById('toggle-pwd');
  const pwdInput = document.getElementById('password');
  const emailInput = document.getElementById('email');

  function showAlert(msg, isError = true) {
    alertBox.textContent = msg;
    alertBox.className = `auth-alert ${isError ? 'error' : 'success'}`;
  }

  // Toggle password visibility
  togglePwd.addEventListener('click', () => {
    if (pwdInput.type === 'password') {
      pwdInput.type = 'text';
      togglePwd.textContent = 'Hide';
    } else {
      pwdInput.type = 'password';
      togglePwd.textContent = 'Show';
    }
  });

  // Google OAuth
  btnGoogle.addEventListener('click', async () => {
    try {
      btnGoogle.disabled = true;
      btnGoogle.textContent = 'Connecting...';
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/index.html`
        }
      });
      if (error) throw error;
    } catch (error) {
      showAlert(error.message);
      btnGoogle.disabled = false;
      btnGoogle.innerHTML = `...`; // Restore standard content here if needed, omitted for brevity
    }
  });

  // Email Login
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = emailInput.value.trim();
    const password = pwdInput.value;

    btnSubmit.disabled = true;
    btnSubmit.textContent = 'Signing in...';
    alertBox.style.display = 'none';

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) throw error;
      
      // Redirect to Streamlit App upon successful login
      // Local Streamlit runs on 8501
      window.location.href = 'http://localhost:8501/';
    } catch (error) {
      showAlert(error.message);
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Sign In';
    }
  });
});
