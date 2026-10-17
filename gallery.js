(() => {
  const lightbox = document.getElementById('lightbox');
  const lightboxInner = lightbox.querySelector('.lightbox-inner');
  const closeBtn = lightbox.querySelector('.close-button');

  let allItems = [];
  let currentIndex = -1;

  function collectItems() {
    allItems = [];
    document.querySelectorAll('.extracted-card, .art-card').forEach(card => {
      allItems.push(card);
    });
  }

  function showItem(index) {
    if (index < 0 || index >= allItems.length) return;
    currentIndex = index;
    const card = allItems[index];
    card.click();
  }

  function openLightbox(card, contentHTML) {
    const idx = allItems.indexOf(card);
    if (idx !== -1) currentIndex = idx;

    const hasPrev = currentIndex > 0;
    const hasNext = currentIndex < allItems.length - 1;

    lightboxInner.innerHTML = `
      ${hasPrev ? '<button class="lb-prev" type="button" aria-label="Previous doodle"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 4l-8 8 8 8"/></svg></button>' : ''}
      ${contentHTML}
      ${hasNext ? '<button class="lb-next" type="button" aria-label="Next doodle"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4l8 8-8 8"/></svg></button>' : ''}
    `;

    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
    closeBtn.focus();

    const prevBtn = lightboxInner.querySelector('.lb-prev');
    const nextBtn = lightboxInner.querySelector('.lb-next');
    if (prevBtn) prevBtn.addEventListener('click', e => { e.stopPropagation(); showItem(currentIndex - 1); });
    if (nextBtn) nextBtn.addEventListener('click', e => { e.stopPropagation(); showItem(currentIndex + 1); });
  }

  function closeLightbox() {
    lightbox.hidden = true;
    document.body.style.overflow = '';
  }

  closeBtn.addEventListener('click', closeLightbox);

  lightbox.addEventListener('mousedown', e => {
    if (e.target === lightbox) closeLightbox();
  });

  document.addEventListener('keydown', e => {
    if (lightbox.hidden) return;
    if (e.key === 'Escape') closeLightbox();
    if (e.key === 'ArrowLeft' && currentIndex > 0) showItem(currentIndex - 1);
    if (e.key === 'ArrowRight' && currentIndex < allItems.length - 1) showItem(currentIndex + 1);
  });

  // ─── Logo Concepts ───

  document.querySelectorAll('.logo-gallery .art-card').forEach(card => {
    card.addEventListener('click', () => {
      const title = card.dataset.title;
      const category = card.dataset.category;
      const year = card.dataset.year;
      const index = card.dataset.index;
      const svg = card.querySelector('.doodle');
      const clone = svg.cloneNode(true);
      const contentHTML = `
        <div class="lightbox-paper">${clone.outerHTML}</div>
        <div class="lightbox-caption">
          <span>${index}</span>
          <div>
            <h2>${title}</h2>
            <p>${category} · logo concept · ${year}</p>
          </div>
        </div>
      `;
      collectItems();
      openLightbox(card, contentHTML);
    });
  });

  // ─── Extracted Doodles ───

  const extractedGallery = document.getElementById('extracted-gallery');
  const sheetFilterContainer = document.querySelector('.doodle-sheet-filter');

  if (extractedGallery) {
    fetch('doodles.json')
      .then(r => r.json())
      .then(doodles => {
        const sheets = [...new Set(doodles.map(d => d.sheet))];
        sheets.forEach(sheet => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.dataset.sheet = sheet;
          const count = doodles.filter(d => d.sheet === sheet).length;
          btn.textContent = sheet.replace('IMG_', '').replace('IMG_', '') + ` (${count})`;
          sheetFilterContainer.appendChild(btn);
        });

        let activeSheet = 'all';

        function renderDoodles() {
          extractedGallery.innerHTML = '';
          const visible = activeSheet === 'all'
            ? doodles
            : doodles.filter(d => d.sheet === activeSheet);

          visible.forEach((d, i) => {
            const card = document.createElement('button');
            card.className = 'extracted-card';
            card.type = 'button';
            card.setAttribute('aria-label', `View doodle ${d.sheet} #${d.index}`);
            card.innerHTML = `
              <img src="images/doodles/${d.file}" alt="Extracted doodle ${d.index} from ${d.sheet}" loading="lazy">
              <span class="card-label">${d.sheet.replace('IMG_', '')} #${d.index}</span>
            `;
            card.addEventListener('click', () => {
              const contentHTML = `
                <div class="lightbox-paper" style="aspect-ratio: auto; background: var(--card-bg); display: flex; align-items: center; justify-content: center; padding: 2rem;">
                  <img class="lightbox-photo" src="images/doodles/${d.file}" alt="Doodle ${d.index} from ${d.sheet}" style="max-height: 70vh; object-fit: contain;">
                </div>
                <div class="lightbox-caption">
                  <span>${String(i + 1).padStart(3, '0')}</span>
                  <div>
                    <h2>${d.sheet.replace('IMG_', 'Sheet ')} — Doodle ${d.index}</h2>
                    <p>extracted · blue pen on paper · 2026</p>
                  </div>
                </div>
              `;
              collectItems();
              openLightbox(card, contentHTML);
            });
            extractedGallery.appendChild(card);
          });
        }

        sheetFilterContainer.addEventListener('click', e => {
          const btn = e.target.closest('button');
          if (!btn) return;
          activeSheet = btn.dataset.sheet;
          sheetFilterContainer.querySelectorAll('button').forEach(b =>
            b.classList.toggle('active', b.dataset.sheet === activeSheet)
          );
          renderDoodles();
        });

        renderDoodles();
      })
      .catch(err => {
        extractedGallery.innerHTML = '<p style="color: var(--text-dim); padding: 2rem;">Could not load doodles. Run the extraction script first.</p>';
      });
  }
})();
