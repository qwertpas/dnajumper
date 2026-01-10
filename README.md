# Gordon Jumper - Project Website

A minimal GitHub Pages site for showcasing robotics project videos.

## Quick Deploy

1. Create a new GitHub repository (e.g., `gordonjumper` or `your-username.github.io`)

2. Push this code:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

3. Enable GitHub Pages:
   - Go to repo **Settings** → **Pages**
   - Source: Deploy from branch
   - Branch: `main` / `(root)`
   - Save

4. Your site will be live at: `https://YOUR_USERNAME.github.io/REPO_NAME/`

## Adding Videos

### Option 1: Local video files
Create a `videos/` folder and add your `.mp4` files, then update the HTML:
```html
<video controls src="videos/my-video.mp4"></video>
```

### Option 2: YouTube embed
```html
<iframe src="https://www.youtube.com/embed/VIDEO_ID" allowfullscreen></iframe>
```

> **Note:** GitHub has a 100MB file size limit. For larger videos, use YouTube or another video host.
