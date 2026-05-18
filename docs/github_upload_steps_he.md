# העלאה ל-GitHub

ה־repo המקומי כבר מוכן עם קומיט ראשון.

## 1. לפתוח Repository חדש ב-GitHub

באתר GitHub:

1. ללחוץ `New repository`.
2. שם מומלץ:

```text
acoustic-drone-detection
```

3. לבחור `Public` אם זה לתיק עבודות.
4. לא לסמן `Add README`, כי README כבר קיים מקומית.
5. ליצור את ה־repo.

## 2. לחבר את התיקייה המקומית ל-GitHub

להחליף את `YOUR_USER` בשם המשתמש שלך:

```bash
git remote add origin https://github.com/YOUR_USER/acoustic-drone-detection.git
git push -u origin main
```

## 3. מה לא עולה ל-GitHub

ה־`.gitignore` מונע העלאה של:

- קבצי אודיו גולמיים,
- תיקיות דאטה גדולות,
- מודלים מאומנים כבדים,
- feature caches,
- רוב תוצרי results הגדולים.

מה שכן עולה:

- קוד,
- README,
- white paper,
- גרפים נבחרים,
- הסבר שימוש ב־AI,
- DATA.md,
- MODEL_CARD.md.

## 4. קישור לקורות חיים

אחרי ההעלאה, לשים בקורות החיים:

```text
Project: Acoustic Drone Detection
GitHub: https://github.com/YOUR_USER/acoustic-drone-detection
```

אפשר גם לשים טקסט לחיץ:

```text
Acoustic Drone Detection - GitHub Project
```
