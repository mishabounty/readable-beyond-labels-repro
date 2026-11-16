"""Перенос существующих приложений и карточек в LaTeX без изменения результатов."""
from pathlib import Path
import re
root=Path(__file__).resolve().parent

def esc(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','≈':r'$\approx$', '⁴':r'$^4$', '√':r'$\sqrt{}$','φ':r'$\varphi$','ᵢ':r'$_i$','→':r'$\rightarrow$','−':'-','±':r'$\pm$','×':r'$\times$','≤':r'$\leq$','≥':r'$\geq$'}
    return ''.join(mapping.get(c,c) for c in s)

def inline(s):
    tokens=[]
    def keep(x):
        tokens.append(x);return f'ZZTOKEN{len(tokens)-1}ZZ'
    def link(m):
        label,url=m.groups()
        if url.startswith('http'):
            return keep(r'\href{'+url.replace('%',r'\%').replace('#',r'\#')+'}{'+esc(label)+'}')
        return esc(label)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)
    s=re.sub(r'`([^`]+)`',lambda m:keep(r'\texttt{'+esc(m[1]).replace(chr(34), r'\string'+chr(34))+'}'),s)
    s=re.sub(r'\*\*([^*]+)\*\*',lambda m:keep(r'\textbf{'+esc(m[1])+'}'),s)
    s=esc(s)
    for i,t in enumerate(tokens):s=s.replace(f'ZZTOKEN{i}ZZ',t)
    return s

def convert(text,catalog=False):
    lines=text.splitlines();out=[];i=0
    if catalog:out=[r'\section{Каталог свидетельств: 55 работ}']
    while i<len(lines):
        s=lines[i].strip();i+=1
        if not s:out.append('');continue
        if s.startswith('# '):continue
        if s.startswith('## '):
            title=s[3:]
            if catalog:
                space=24 if title.startswith('55.') else 9
                out.append(r'\Needspace{'+str(space)+r'\baselineskip}\subsection*{'+inline(title)+'}')
            else:
                title=re.sub(r'^[A-Z]\.\s*','',title)
                out.append(r'\section{'+inline(title)+'}')
            continue
        if s.startswith('### '):
            title=re.sub(r'^[A-Z]\.\d+\.\s*','',s[4:])
            out.append(r'\subsection{'+inline(title)+'}');continue
        if s.startswith('|'):
            rows=[s]
            while i<len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip());i+=1
            cells=[[x.strip() for x in r.strip('|').split('|')] for r in rows]
            cells=[r for r in cells if not all(re.fullmatch(r'[-: ]+',x) for x in r)]
            n=len(cells[0]);width=(.99-.027*n)/n
            widths=[width]*n
            if cells[0][0]=="Код":widths=[.10,.50,.16,.12]
            spec=''.join('>{\\raggedright\\arraybackslash}p{'+str(round(w,3))+r'\linewidth}' for w in widths)
            out.append(r'{\small\begin{longtable}{'+spec+'}'+r'\toprule')
            out.append(' & '.join(inline(c) for c in cells[0])+r'\\\midrule\endhead')
            for row in cells[1:]:out.append(' & '.join(inline(c) for c in row)+r'\\')
            out.append(r'\bottomrule\end{longtable}}');continue
        if s.startswith('- '):s='• '+s[2:]
        out.append(inline(s))
    return '\n'.join(out)+'\n'
for src,dst,cat in [('supplement_ru.md','supplement.tex',False),('literature_evidence.md','catalog.tex',True)]:
    (root/dst).write_text(convert((root.parent/src).read_text(),cat))
