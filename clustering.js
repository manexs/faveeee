document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked button and corresponding tab
            this.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
    
    // Handle file upload
    const uploadForm = document.getElementById('upload-form');
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('dataset-file');
        if (!fileInput.files.length) {
            showError('Please select a file to upload');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        fetch('/upload_dataset', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showDatasetInfo(data.filename);
                showClusteringSection();
                
                // NEW: Load user dropdown after successful upload
                loadUserDropdown(data.filename);
            } else {
                showError(data.error || 'Error uploading file');
            }
        })
        .catch(error => {
            showError('Error: ' + error.message);
        });
    });
    
    // Handle dataset generation
    const generateForm = document.getElementById('generate-form');
    generateForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const nUsers = document.getElementById('n-users').value;
        
        const formData = new FormData();
        formData.append('n_users', nUsers);
        
        fetch('/generate_dataset', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showDatasetInfo(data.filename);
                showClusteringSection();
                
                // NEW: Load user dropdown after successful generation
                loadUserDropdown(data.filename);
            } else {
                showError(data.error || 'Error generating dataset');
            }
        })
        .catch(error => {
            showError('Error: ' + error.message);
        });
    });
    
    // Handle clustering
    const clusteringForm = document.getElementById('clustering-form');
    clusteringForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const nClusters = document.getElementById('n-clusters').value;
        const findOptimal = document.getElementById('find-optimal').checked;
        const filename = document.getElementById('dataset-filename').textContent;
        
        const formData = new FormData();
        formData.append('n_clusters', nClusters);
        formData.append('find_optimal', findOptimal);
        formData.append('filename', filename);
        
        document.getElementById('clustering-progress').classList.remove('hidden');
        document.getElementById('clustering-results').classList.add('hidden');
        
        fetch('/perform_clustering', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('clustering-progress').classList.add('hidden');
            
            if (data.success) {
                showClusteringResults(data);
                
                // NEW: Load user dropdown with clustered data
                loadUserDropdown(data.clustered_filename || filename);
            } else {
                showError(data.error || 'Error performing clustering');
            }
        })
        .catch(error => {
            document.getElementById('clustering-progress').classList.add('hidden');
            showError('Error: ' + error.message);
        });
    });
    
    // NEW: Add function to populate user dropdown after clustering
    function loadUserDropdown(filename) {
        // Show the user selection section
        document.getElementById('user-selection-section').style.display = 'block';
        
        // Fetch users from the dataset
        fetch(`/get_users?filename=${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const dropdown = document.getElementById('user-dropdown');
                    dropdown.innerHTML = '<option value="">Select a user...</option>';
                    
                    // Add users to dropdown
                    data.users.forEach(user => {
                        const option = document.createElement('option');
                        option.value = user.user_id;
                        option.textContent = `User #${user.user_id} (${user.age}, ${user.location})`;
                        dropdown.appendChild(option);
                    });
                }
            })
            .catch(error => console.error('Error loading users:', error));
    }

    // NEW: Add event listener for the "Chat as Selected User" button
    const loginButton = document.getElementById('login-as-user');
    if (loginButton) {
        loginButton.addEventListener('click', function() {
            const userId = document.getElementById('user-dropdown').value;
            
            if (!userId) {
                alert('Please select a user first');
                return;
            }
            
            // Set the user context and redirect to chat
            fetch('/set_user_context', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_id: userId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    window.location.href = '/chat';
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => console.error('Error setting user context:', error));
        });
    }
    
    // Helper functions
    function showDatasetInfo(filename) {
        const datasetInfo = document.getElementById('dataset-info');
        const datasetFilename = document.getElementById('dataset-filename');
        
        datasetFilename.textContent = filename;
        datasetInfo.classList.remove('hidden');
    }
    
    function showClusteringSection() {
        const clusteringSection = document.getElementById('clustering-section');
        clusteringSection.classList.remove('hidden');
    }
    
    function showClusteringResults(data) {
        const resultsDiv = document.getElementById('clustering-results');
        const silhouetteScore = document.getElementById('silhouette-score');
        const pcaViz = document.getElementById('pca-viz');
        const tsneViz = document.getElementById('tsne-viz');
        const clusterProfiles = document.getElementById('cluster-profiles');
        
        // Show silhouette score
        silhouetteScore.textContent = `Silhouette score: ${data.silhouette_avg.toFixed(3)}`;
        
        // Show visualizations
        if (data.visualizations) {
            if (data.visualizations.pca) {
                pcaViz.src = `data:image/png;base64,${data.visualizations.pca}`;
            }
            
            if (data.visualizations.tsne) {
                tsneViz.src = `data:image/png;base64,${data.visualizations.tsne}`;
            }
        }
        
        // Show cluster profiles
        clusterProfiles.innerHTML = '';
        
        if (data.cluster_profiles) {
            data.cluster_profiles.forEach(profile => {
                const profileDiv = document.createElement('div');
                profileDiv.className = 'cluster-profile';
                
                let profileHTML = `
                    <h4>Cluster ${profile.cluster_id}</h4>
                    <div class="stats">
                        <p><strong>Size:</strong> ${profile.size} users (${profile.percentage.toFixed(1)}%)</p>
                        <p><strong>Average Age:</strong> ${profile.avg_age.toFixed(1)}</p>
                    </div>
                    
                    <h5>Top Locations:</h5>
                    <ul>
                `;
                
                profile.top_locations.forEach(loc => {
                    profileHTML += `<li>${loc[0]}: ${(loc[1] * 100).toFixed(1)}%</li>`;
                });
                
                profileHTML += `
                    </ul>
                    
                    <h5>Top Interests:</h5>
                    <ul>
                `;
                
                profile.top_interests.forEach(interest => {
                    profileHTML += `<li>${interest[0]}: ${(interest[1] * 100).toFixed(1)}%</li>`;
                });
                
                profileHTML += `
                    </ul>
                    
                    <h5>Top Events:</h5>
                    <ul>
                `;
                
                profile.top_events.forEach(event => {
                    profileHTML += `<li>${event[0]}: ${(event[1] * 100).toFixed(1)}%</li>`;
                });
                
                profileHTML += `</ul>`;
                
                profileDiv.innerHTML = profileHTML;
                clusterProfiles.appendChild(profileDiv);
            });
        }
        
        resultsDiv.classList.remove('hidden');
    }
    
    function showError(message) {
        // Create error alert
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert error';
        errorAlert.textContent = message;
        
        // Add it to the page
        const dataSection = document.getElementById('data-section');
        dataSection.appendChild(errorAlert);
        
        // Remove it after 5 seconds
        setTimeout(() => {
            errorAlert.remove();
        }, 5000);
    }
});