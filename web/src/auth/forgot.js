import { supabase } from './supabase.js';

document.addEventListener('DOMContentLoaded', () => {
  const forgotForm = document.getElementById('forgot-form');
  const btnSubmit = document.getElementById('btn-submit');
  const alertBox = document.getElementById('alert-box');
  const emailInput = document.getElementById('email');

  function showAlert(msg, isError = true) {
    alertBox.textContent = msg;
    alertBox.className = `auth-alert ${isError ? 'error' : 'success'}`;
    alertBox.style.display = 'block';
  }

  forgotForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = emailInput.value.trim();

    btnSubmit.disabled = true;
    btnSubmit.textContent = 'Sending...';
    alertBox.style.display = 'none';

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password.html`,
      });

      if (error) throw error;
      
      showAlert('Password reset link has been sent to your email.', false);
    } catch (error) {
      showAlert(error.message);
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Send reset link';
    }
  });
});
