// Function to preview the uploaded image
function uploadImage() {
  const fileInput = document.getElementById("image-upload");
  const previewImage = document.getElementById("uploaded-image");

  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      previewImage.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
}

// Function to enhance the image using the backend (AI model)
function enhanceImage() {
  const fileInput = document.getElementById("image-upload");
  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  fetch("/enhance", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.image_url && data.original_url) {
        // Display the comparison panel
        const comparisonPanel = document.getElementById("comparison-panel");
        comparisonPanel.style.display = "block";

        // Display original and enhanced images
        document.getElementById("original-image").src = data.original_url;
        document.getElementById("enhanced-image").src = data.image_url;
      } else {
        alert("Error enhancing image");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Error enhancing image");
    });
}
