import { supabase } from './supabase.js';

document.addEventListener('DOMContentLoaded', () => {
  const signupForm = document.getElementById('signup-form');
  const btnGoogle = document.getElementById('btn-google');
  const btnSubmit = document.getElementById('btn-submit');
  const alertBox = document.getElementById('alert-box');
  const togglePwd = document.getElementById('toggle-pwd');
  const pwdInput = document.getElementById('password');
  const confirmPwdInput = document.getElementById('confirm-password');
  const emailInput = document.getElementById('email');
  const nameInput = document.getElementById('fullname');

  function showAlert(msg, isError = true) {
    alertBox.textContent = msg;
    alertBox.className = `auth-alert ${isError ? 'error' : 'success'}`;
    alertBox.style.display = 'block';
  }

  // Toggle password visibility
  togglePwd.addEventListener('click', () => {
    if (pwdInput.type === 'password') {
      pwdInput.type = 'text';
      confirmPwdInput.type = 'text';
      togglePwd.textContent = 'Hide';
    } else {
      pwdInput.type = 'password';
      confirmPwdInput.type = 'password';
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
    }
  });

  // Email Signup
  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = emailInput.value.trim();
    const password = pwdInput.value;
    const confirmPassword = confirmPwdInput.value;
    const fullName = nameInput.value.trim();

    if (password !== confirmPassword) {
      showAlert('Passwords do not match');
      return;
    }

    if (password.length < 6) {
      showAlert('Password must be at least 6 characters');
      return;
    }

    btnSubmit.disabled = true;
    btnSubmit.textContent = 'Creating account...';
    alertBox.style.display = 'none';

    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: fullName,
          }
        }
      });

      if (error) throw error;
      
      // If email confirmation is required, Supabase handles it.
      if (data?.user && data.user.identities && data.user.identities.length === 0) {
        showAlert('An account with this email already exists.');
        btnSubmit.disabled = false;
        btnSubmit.textContent = 'Create account';
        return;
      }
      
      showAlert('Account created! You may now sign in (or check your email for confirmation).', false);
      
      setTimeout(() => {
          // Redirect to login or automatically sign in based on Supabase settings
          // usually auto sign-in works if email confirmation is disabled
          window.location.href = 'https://krshz-sudo-sih-3-0-guiapp-0esz5j.streamlit.app/';
      }, 2000);

    } catch (error) {
      showAlert(error.message);
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Create account';
    }
  });
});
