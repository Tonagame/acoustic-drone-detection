# תקציר לפרויקט בקורות חיים

## מערכת זיהוי רחפנים אקוסטית מבוססת למידת מכונה ועיבוד אות

פיתוח מערכת מחקרית לזיהוי רחפנים לפי שמע, עם דגש על הפחתת התרעות שווא ממנועים, כלי רכב, טנקים, קהל ורעשי סביבה.

המערכת התפתחה ממודל CNN בסיסי למערכת מרובת רכיבים:

- חמישה מודלי CNN מומחים, כל אחד על תחום סינון שונה של האודיו.
- מאפייני DSP לזיהוי מבנה הרמוני של מנועים וכלי רכב.
- מודל pitch/periodicity pretrained לזיהוי מחזוריות קולית.
- שכבת learned ML fusion שמחברת את כל הראיות להחלטת drone / no-drone.
- בדיקות benchmark מול DADS ו־FSD50K, כולל רעשי מנוע/רכב אמיתיים ומיקסים ב־SNR משתנה.

תוצאה נוכחית בבנצ'מרק:

| מדד | תוצאה |
|---|---:|
| זיהוי רחפן נקי | 99.20% |
| זיהוי רחפן עם רעש FSD50K אמיתי | 91.05% |
| התרעות שווא בבנצ'מרק שלילי | 0.00% |

הפרויקט כולל:

- קוד Python/PyTorch לאימון, הערכה וגרפים.
- white paper הנדסי מלא.
- גרפים להשוואת איטרציות.
- סימולציית microphone array / beamforming ראשונית.
- תיעוד איך שימוש ב־AI עזר בתהליך המחקר, הדיבוג והכתיבה.

שורת קורות חיים מוצעת:

```text
Developed a passive acoustic drone detection prototype using multi-view CNN specialists, harmonic DSP, pretrained pitch estimation, and learned ML fusion; achieved 91% mixed real-noise recall with near-zero false alarms on benchmark tests.
```
