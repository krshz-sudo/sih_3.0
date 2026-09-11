import { supabase } from './supabase.js';

document.addEventListener('DOMContentLoaded', async () => {
  const resetForm = document.getElementById('reset-form');
  const btnSubmit = document.getElementById('btn-submit');
  const alertBox = document.getElementById('alert-box');
  const pwdInput = document.getElementById('password');
  const togglePwd = document.getElementById('toggle-pwd');

  function showAlert(msg, isError = true) {
    alertBox.textContent = msg;
    alertBox.className = `auth-alert ${isError ? 'error' : 'success'}`;
    alertBox.style.display = 'block';
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

  // Supabase automatically parses the URL hash containing the access token on load
  // If the user lands here without a valid recovery hash, we should warn them.
  const { data: { session }, error } = await supabase.auth.getSession();
  
  // Note: if Supabase client successfully extracted the recovery token from URL,
  // we will have a valid session to update the user.
  
  resetForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const newPassword = pwdInput.value;

    if (newPassword.length < 6) {
      showAlert('Password must be at least 6 characters');
      return;
    }

    btnSubmit.disabled = true;
    btnSubmit.textContent = 'Updating...';
    alertBox.style.display = 'none';

    try {
      const { error } = await supabase.auth.updateUser({
        password: newPassword
      });

      if (error) throw error;
      
      showAlert('Password successfully updated!', false);
      
      setTimeout(() => {
        window.location.href = 'http://localhost:8501/';
      }, 2000);
      
    } catch (error) {
      showAlert(error.message);
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Update password';
    }
  });
});
