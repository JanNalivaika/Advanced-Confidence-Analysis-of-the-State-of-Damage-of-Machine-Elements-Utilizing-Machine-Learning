\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{FZGdasa}[2013/01/01 LaTeX2e fuer FZG DA und SA]

\newif\if@entwurf\@entwurffalse
\DeclareOption{entwurf}{\@entwurftrue}
\newif\if@english\@englishfalse
\DeclareOption{english}{\@englishtrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ExecuteOptions{english}
\ProcessOptions\relax%

\LoadClass[11pt,a4paper,fleqn]{report}

%Verbessertes Float-Handling mit Schalter H (gleiche Position wie im Skript)
\RequirePackage{float}
%verhindert Figuren vor ihrer ersten Referenz
\RequirePackage{flafter}
%kann eine Barriere für Float-Umgebungen erzeugen mit \FloatBarrier
\RequirePackage{placeins}
%%verbessertes Beschriften mir div. Optionen
%\RequirePackage{caption}
%%zusaetzlihce Symbole und Schriften (ams: american mathematical soc)
%%\RequirePackage{amssymb}
\RequirePackage[fleqn]{amsmath}
%\RequirePackage{amstext}
\RequirePackage{amsfonts}
%\RequirePackage{amsbsy}
%\RequirePackage{amscd}
%\RequirePackage{latexsym}

%\RequirePackage{eurosym}
%drehen von Text, Tabellen, Seiten
%\RequirePackage{rotating}
%%including graphics files, rotating parts of a page, and scaling parts of a page
%\RequirePackage{graphicx}
%Für den Index
%\RequirePackage{makeidx}
%Erleichterungen fürs Deutsche inkl Silbentrennung
% \RequirePackage{ngerman}
\if@english
	\PassOptionsToPackage{ngerman, main=english}{babel}%
\else
	\PassOptionsToPackage{english, main=ngerman}{babel}%
\fi
\RequirePackage{babel}%
\RequirePackage{a4}
%direkte Eingabe von Umlauten mit Angabe von Schriftsatz
%in Kombination mit 'german' sind jetzt ä direkt erlaubt!
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[hyperindex,bookmarksopen,bookmarksnumbered,colorlinks=true,citecolor=LinkColor,filecolor=LinkColor,linkcolor=LinkColor,%
      urlcolor=LinkColor]{hyperref}
%\RequirePackage[pdftex,hyperindex,colorlinks=false,bookmarksopen,bookmarksnumbered,pdftitle={Titel},pdfauthor={He}]{hyperref}
%\RequirePackage{subfigure}
\RequirePackage{caption}
\RequirePackage{fancyhdr}
\RequirePackage{afterpage}
\RequirePackage{titlesec}
\RequirePackage[titles]{tocloft}
\RequirePackage{fleqn}
\RequirePackage{lastpage}
\RequirePackage{longtable}
\RequirePackage[final]{pdfpages}
%\RequirePackage{wrapfig}
\RequirePackage{multirow} % fuer Mehrfachzeilen in Tabellen
\RequirePackage{mathcomp} % aufrechte Symbole in Mathmode: \tcmu, \tccelcius, \tcohm, \tcdigitoldstyle, \tcperthousand
\RequirePackage{gensymb}  % \micro fuer nices mue in Textmode
\RequirePackage{paralist} % Verbesserte/Anpassbare Aufzaehlungen, z.B.: Umgebung \compactitem
\RequirePackage{trfsigns} % fuer \fourier \laplace
\RequirePackage{mathdots} % \vdots in Math Mode korrigieren (rund)
\RequirePackage[sortbib]{FZGdef}
\RequirePackage{ifthen}
\if@entwurf
% \RequirePackage{datetime}
 \PassOptionsToPackage{\CurrentOption}{FZGdef}
 
 \newcommand{\CmpTime}{\textcolor{red}{\textbf{\textit{Entwurf}} Stand: \CompiledTime}}

\fi
%PAGE LAYOUT
% Latex hat per default links und oben 1 inch Rand!!
\setlength{\topmargin}{-25mm}
\setlength{\headheight}{5.mm}
\setlength{\headsep}{+6.mm}
\setlength{\textheight}{240mm}
\setlength{\footskip}{+20.mm}
\setlength{\textwidth}{161mm} %!!!
\setlength{\oddsidemargin}{0mm}
\setlength{\evensidemargin}{0mm}
\setlength{\parindent}{0pt}
\setlength{\paperwidth}{210mm}
\setlength{\paperheight}{297mm}
 
% keeps the distance between paragraphs constant
\setlength{\parskip}{1ex plus 0.0ex minus 0.0ex}
 
\renewcommand{\floatpagefraction}{0.9}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\textfloatsep}{5mm}
\renewcommand{\baselinestretch}{1.3}

%------ BEGINN AUS SEITENSTIL.TEX -------------
%Kopfzeilen
% erste Seite
\fancypagestyle{erste}{
\renewcommand{\headheight}{3.25cm}
\lhead{
\includegraphics[height=13.15mm]{_ltx/FZGLogo}
}
\chead{%
	\begin{otherlanguage}{ngerman}%
		\begin{minipage}[b][12mm][c]{\textwidth}%
			\centering
			\color{fzgblau}\bfseries\footnotesize
			Lehrstuhl für Maschinenelemente\\
			Forschungsstelle für Zahnräder und Getriebesysteme\\
			Prof. Dr.-Ing. K. Stahl%
		\end{minipage}%
	\end{otherlanguage}%
%\includegraphics[width=1.00\textwidth]{_ltx/Kopf.pdf}
}
\rhead{
% Umrandetes TUM-Logo (nicht mehr erlaubt) % Todo: Option zur Verwendung dieses Logos einfügen oder samt Dateien gleich entfernen
% \includegraphics[height=12.85mm]{_ltx/TULogo}%
%
% Ausgefülltes TUM-Logo % Todo: Höhe TUM-Logo ca. 88% von Höhe FZG-Logo (wie in Präsentationsvorlage)? Auf Überarbeitung Word-Vorlage warten.
\includegraphics[height=12.85mm]{_ltx/2015_Logo_TUM_RGB.pdf}%
}
\lfoot{}
\cfoot{}
\rfoot{}
\renewcommand{\headrulewidth}{1pt}
\renewcommand{\footrulewidth}{0pt}
\renewcommand{\headrule}{\vskip1pt \color{fzgblau}\hrule width\headwidth height\headrulewidth \vskip-\headrulewidth}
}

% Alle anderen Seiten
% Todo: twoside-Support
\AtBeginDocument{%
    \renewcommand{\sectionmark}[1]{\markright{\thesection\ \ #1}{}}%%% twoside
    %\renewcommand\chaptermark[1]{%
    %	\markboth {\MakeUppercase{%
    %			\ifnum \c@secnumdepth >\m@ne
    %			\@chapapp\ \thechapter \ %
    %			\fi
    %			#1}}{}}
    \renewcommand{\chaptermark}[1]{\markboth{\thechapter\ \ #1}{}}%
}%
\fancypagestyle{plain}{
\renewcommand{\headheight}{2.15cm}
% \lhead[\color{fzgblau}\textbf{\thepage}]{\color{fzgblau}\textbf{\nouppercase{\leftmark}}}
\lhead[\textbf{\thepage}]{\textbf{\nouppercase{\leftmark}}}
\chead{}
% \rhead[\color{fzgblau}\textbf{\nouppercase{\leftmark}}]{\color{fzgblau}\textbf{\thepage}}
\rhead[\textbf{\nouppercase{\leftmark}}]{\textbf{\thepage}}
\lfoot{}
\cfoot{\if@entwurf%
        \CmpTime
        \fi}
\rfoot{}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
% \renewcommand{\headrule}{\color{fzgblau}\hrule width\headwidth height\headrulewidth \vskip-\headrulewidth}
\renewcommand{\headrule}{\hrule width\headwidth height\headrulewidth \vskip-\headrulewidth}
}
%
\thispagestyle{erste}
\pagestyle{plain}
%
\setcounter{secnumdepth}{4}
\setcounter{tocdepth}{4}
%
%Kapitelüberschriften
\titleformat{\chapter}%
  {\normalfont\bfseries}%
  {\thechapter}%
  {1em}%
  {}%
\titlespacing{\chapter}{0pt}{-22.5pt}{0pt}% Todo
%
\titleformat{\section}%
  {\normalfont\bfseries}%
  {\thesection}%
  {1em}%
  {}%
\titlespacing{\section}{0pt}{10pt}{0pt}
%
\titleformat{\subsection}%
  {\normalfont\bfseries}%
  {\thesubsection}%
  {1em}%
  {}%
  \titlespacing{\subsection}{0pt}{7pt}{0pt}
%
\titleformat{\subsubsection}%
  {\normalfont\bfseries}%
  {\thesubsubsection}%
  {1em}%
  {}%
  \titlespacing{\subsubsection}{0pt}{4pt}{0pt}

%Bildunterschrift
\captionsetup{font=small,labelfont=bf,format=hang,singlelinecheck=false}
%
% \renewcommand{\figurename}{Bild}
% \renewcommand{\contentsname}{Inhaltsverzeichnis}
%
%Inhaltsverzeichnis formatieren tocloft
%\renewcommand{\cfttoctitlefont}{\normalfont\bfseries}
%\renewcommand{\cftbeforetoctitleskip}{0pt}
%\renewcommand{\cftaftertoctitleskip}{0pt}
%%\renewcommand{\cfttoctitlepresnum}{}
%%\renewcommand{\cfttoctitleaftersnum}{}
%\renewcommand{\cftbeforetoctitleskip}{-17.75pt}
% Wird durch die Option titles von tocloft angepasst
% https://tex.stackexchange.com/questions/445952/tocloft-change-font-size-of-list-of-figures
%
\cftsetindents{chapter}{0cm}{1.4cm}
\renewcommand{\cftchapdotsep}{1.5}
%\renewcommand{\cftchapfont}{\normalfont}
\renewcommand{\cftchapleader}{\normalfont\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftchappagefont}{\normalfont}
\renewcommand{\cftchappresnum}{}
\renewcommand{\cftchapaftersnum}{}
\renewcommand{\cftbeforechapskip}{4mm}

\cftsetindents{section}{0.6cm}{1.4cm}
\renewcommand{\cftsecdotsep}{1.5}
\renewcommand{\cftsecfont}{\normalfont}
\renewcommand{\cftsecleader}{\normalfont\cftdotfill{\cftsubsecdotsep}}
\renewcommand{\cftsecpagefont}{\normalfont}
\renewcommand{\cftsecpresnum}{}
\renewcommand{\cftsecaftersnum}{}
\renewcommand{\cftbeforesecskip}{0pt}

\cftsetindents{subsection}{1.2cm}{1.4cm}
\renewcommand{\cftsubsecdotsep}{1.5}
\renewcommand{\cftsubsecfont}{\normalfont}
\renewcommand{\cftsubsecleader}{\normalfont\cftdotfill{\cftsubsubsecdotsep}}
\renewcommand{\cftsubsecpagefont}{\normalfont}
\renewcommand{\cftsubsecpresnum}{}
\renewcommand{\cftsubsecaftersnum}{}
\renewcommand{\cftbeforesubsecskip}{0pt}

\cftsetindents{subsubsection}{1.8cm}{1.4cm}
\renewcommand{\cftsubsubsecdotsep}{1.5}
\renewcommand{\cftsubsubsecfont}{\normalfont}
\renewcommand{\cftsubsubsecleader}{\normalfont\cftdotfill{\cftsubsubsecdotsep}}
\renewcommand{\cftsubsubsecpagefont}{\normalfont}
\renewcommand{\cftsubsubsecpresnum}{}
\renewcommand{\cftsubsubsecaftersnum}{}
\renewcommand{\cftbeforesubsubsecskip}{0pt}%

\cftsetindents{paragraph}{2.4cm}{1.8cm}
\renewcommand{\cftparadotsep}{1.5}
\renewcommand{\cftparafont}{\normalfont}
\renewcommand{\cftparaleader}{\normalfont\cftdotfill{\cftparadotsep}}
\renewcommand{\cftparapagefont}{\normalfont}
\renewcommand{\cftparapresnum}{}
\renewcommand{\cftparaaftersnum}{}
\renewcommand{\cftbeforeparaskip}{0pt}%
\floatplacement{figure}{H}
\floatplacement{table}{H}

%
%Automatische Refernezierung auf Bilder
% \newcommand{\Bildref}[1]{%
% \textbf{Bild \ref{#1}}}%
%Automatische Refernezierung auf Tabellen
% \newcommand{\Tabref}[1]{%
% \textbf{Tabelle \ref{#1}}}%
%Automatische Refernezierung auf Tabellen
% \newcommand{\Glref}[1]{%
% \textbf{Formel (\ref{#1})}}%
%Automatische Refernezierung auf Abschintte
% \newcommand{\Kapref}[2]{%
% \textbf{#1 \ref{#2}}}%
%
%Automatische itemize in einer Bildunterschrift
\newcommand{\Bilditem}[1]{%
\newline \hspace*{#1} $\bullet\ $ }%
%

% Definition des Anhang-Befehles
\newcommand \Anhang{\par
\setcounter{chapter}{0}%
\setcounter{section}{0}%
\setcounter{subsection}{0}%
\setcounter{figure}{0}%
\setcounter{table}{0}%
\renewcommand\thesection{\Alph{chapter}} %
\renewcommand\thefigure{\Alph{chapter}.\arabic{figure}}
\section{Anhang}
\label{sec:Anhang}
}
%-----------------------------------------------------------------------
% Definition Eingabe aus Preamble
%-----------------------------------------------------------------------
\newcommand*{\@Arbeit}{}
\newcommand{\Arbeit}[1]{\global\def\@Arbeit{#1}}
\ifthenelse{\equal{\@Arbeit}{}}{%
   \renewcommand{\@Arbeit}{%
      \ErrMsg{\texttt{$\backslash$Arbeit\{Diplomarbeit\}}}
   }}{}%%
\ifthenelse{\equal{\@Arbeit}{{}}}{%
   \renewcommand{\@Arbeit}{%
      \ErrMsg{\texttt{$\backslash$Arbeit\{Diplomarbeit\}}}
   }}{}%%

\newcommand*{\@Nummer}{}
\newcommand{\Nummer}[1]{\gdef\@Nummer{#1}}
\ifthenelse{\equal{\@Nummer}{}}{%
   \renewcommand{\@Nummer}{%
      \ErrMsg{\texttt{$\backslash$Nummer\{\#\#\#\}}}
  }}{}%%

\newcommand*{\@ThemaDeutsch}{}
\newcommand{\ThemaDeutsch}[1]{\gdef\@ThemaDeutsch{#1}}
\if@english
\else
	\ifthenelse{\equal{\@ThemaDeutsch}{}}{%
	   \renewcommand{\@ThemaDeutsch}{%
		  \ErrMsg{\texttt{$\backslash$Thema\{Thema in Deutsch\}}}
	   }}{}%%
\fi

\newcommand*{\@ThemaEnglisch}{}
\newcommand{\ThemaEnglisch}[1]{\gdef\@ThemaEnglisch{#1}}
\if@english
	% Nur falls Englisch-Option, da bei Semesterarbeit kein Titel in Englisch gefordert
	\ifthenelse{\equal{\@ThemaEnglisch}{}}{%
	   \renewcommand{\@ThemaEnglisch}{%
		  \ErrMsg{\texttt{$\backslash$ThemaEnglisch\{Thema in Englisch\}}}
	   }}{}%%
\else
\fi

\newcommand*{\@ErlangungGrad}{}
\newcommand{\ErlangungGrad}[1]{\gdef\@ErlangungGrad{#1}}
% \ifthenelse{\equal{\@ErlangungGrad}{}}{%
%    \renewcommand{\@ErlangungGrad}{%
%       \ErrMsg{\texttt{$\backslash$ErlangungGrad\{Erlangung Grad\}}}
%    }}{}%%

\newcommand*{\@AutorAnrede}{}
\newcommand{\AutorAnrede}[1]{\gdef\@AutorAnrede{#1}}
\ifthenelse{\equal{\@AutorAnrede}{}}{%
   \renewcommand{\@AutorAnrede}{%
      \ErrMsg{\texttt{$\backslash$AutorAnrede\{Herr oder Frau\}}}
   }}{}%%

\newcommand*{\@Autorname}{}
\newcommand{\Autorname}[2][]{\global\def\@Autorname{%
\ifthenelse{\equal{#1}{}}{}{#1~}#2}}
\ifthenelse{\equal{\@Autorname}{}}{%
   \renewcommand{\@Autorname}{%
      \ErrMsg{\texttt{$\backslash$Autorname[<Titel>]\{Vorname Nachname\}}}
   }}{}%%

\newcommand*{\@AutorMatrNr}{}
\newcommand{\AutorMatrNr}[1]{\gdef\@AutorMatrNr{#1}}
\ifthenelse{\equal{\@AutorMatrNr}{}}{%
   \renewcommand{\@AutorMatrNr}{%
      \ErrMsg{\texttt{$\backslash$AutorMatrNr\{01234567\}}}
   }}{}%%

\newcommand*{\@Betreuername}{}
\newcommand{\Betreuername}[2][]{\global\def\@Betreuername{%
\ifthenelse{\equal{#1}{}}{}{#1~}#2}}%
\ifthenelse{\equal{\@Betreuername}{}}{%
   \renewcommand{\@Betreuername}{%
      \ErrMsg{\texttt{$\backslash$Betreuername[<Titel>]\{Vorname Nachname\}}}
   }}{}%%

\newcommand*{\@Themenstellername}{}
\newcommand{\Themenstellername}[2][]{\global\def\@Themenstellername{%
\ifthenelse{\equal{#1}{}}{}{#1~}#2}}
\ifthenelse{\equal{\@Themenstellername}{}}{%
   \renewcommand{\@Themenstellername}{%
      \ErrMsg{\texttt{$\backslash$Themenstellername[<Titel>]\{Vorname Nachname\}}}
   }}{}%%

\newcommand*{\@ThemenstellerZusatz}{}
\newcommand{\ThemenstellerZusatz}[1]{\global\def\@ThemenstellerZusatz{#1}}
\ifthenelse{\equal{\@ThemenstellerZusatz}{}}{%
   \renewcommand{\@ThemenstellerZusatz}{%
      \ErrMsg{\texttt{$\backslash$ThemenstellerZusatz\{Zusatz\}}}
   }}{}%%

\newcommand*{\@Beginn}{}
\newcommand{\Beginn}[1]{\global\def\@Beginn{#1}}
\ifthenelse{\equal{\@Beginn}{}}{%
   \renewcommand{\@Beginn}{%
      \ErrMsg{\texttt{$\backslash$Beginn\{dd.mm.yyyy.\}}}
   }}{}%%

\newcommand*{\@Ende}{}
\newcommand{\Ende}[1]{\global\def\@Ende{#1}}
\ifthenelse{\equal{\@Ende}{}}{%
   \renewcommand{\@Ende}{%
      \ErrMsg{\texttt{$\backslash$Ende\{dd.mm.yyyy.\}}}
   }}{}%%

\newcommand*{\@Ort}{}
\newcommand{\Ort}[1]{\global\def\@Ort{#1}}
\ifthenelse{\equal{\@Ort}{}}{%
   \renewcommand{\@Ort}{%
      \ErrMsg{\texttt{$\backslash$Ort\{Ort\}}}
   }}{}%%

\newcommand*{\@Aufgabepfad}{}
\newcommand{\Aufgabepfad}[1]{\global\def\@Aufgabepfad{#1}}
%\newcommand*{\@Aufgabe}{}
%\newcommand{\Aufgabe}[1]{\gdef\input{\@Aufgabe{#1}}}
%
% Fallunterscheidung Thema
\newcommand*{\ThemaPrimaersprache}{}%
\newcommand*{\ThemaSekundaersprache}{}%
\if@english
	\renewcommand*{\ThemaPrimaersprache}{\@ThemaEnglisch}%
	\renewcommand*{\ThemaSekundaersprache}{\@ThemaDeutsch}%
\else
	\renewcommand*{\ThemaPrimaersprache}{\@ThemaDeutsch}%
	\renewcommand*{\ThemaSekundaersprache}{\@ThemaEnglisch}%
\fi
%
% Fallunterscheidung Sprache
\addto\captionsngerman{%
	\renewcommand*{\contentsname}{Inhaltsverzeichnis}%
	\gdef\bibname{Literaturverzeichnis}% Todo: \renewcommand not working
	\renewcommand*{\figurename}{Bild}%
}
\addto\captionsenglish{%
	\renewcommand*{\contentsname}{Table of contents}%
	\gdef\bibname{References}% Todo: \renewcommand not working
}%
%
%\newcommand{\profname}{Prof. Dr.-Ing. B.-R. Höhn}  

\renewcommand*\titlepage{%
% \null behebt Problem mit verändertem Abstand bei neueren LaTeX-Versionen (ab 2021?)
% https://tex.stackexchange.com/questions/24919/what-is-null-and-when-do-we-need-to-use-it
% Todo: Genaue Version mit relevanten Änderungen bzw. relevantes Paket suchen
\null
%
\thispagestyle{erste}%
\vspace*{0.35cm}%
\begin{center}
\begin{otherlanguage}{ngerman}%
	\huge\bfseries
	\@Arbeit\\
	\vspace{-0.15cm}%
	Nr. \@Nummer\\
	\vspace*{1.5cm}%
\end{otherlanguage}%
\Large\bfseries
\ThemaPrimaersprache\\
\ifthenelse{\equal{\ThemaSekundaersprache}{}}{}{%
	\vspace*{1.3cm}%
	\if@english
		\begin{otherlanguage}{ngerman}%
			\ThemaSekundaersprache
		\end{otherlanguage}%
	\else
		\begin{otherlanguage}{english}%
			\ThemaSekundaersprache
		\end{otherlanguage}%
	\fi
	\\
}%
\vfill
\begin{otherlanguage}{ngerman}%
\ifthenelse{\equal{\@ErlangungGrad}{}}{}{%
	{\renewcommand{\baselinestretch}{1}\raggedright\large\normalfont
	Wissenschaftliche Arbeit zur Erlangung des Grades\\
	\@ErlangungGrad\\
	an der TUM School of Engineering and Design der Technischen Universität München.%
	\vspace{2cm}\\}%
}%
\normalsize\mdseries\normalfont
\begin{tabular}{@{}l@{\hspace{0.8cm}}l@{}}
Eingereicht von:			& \@Autorname\\[0.05cm]
							& Matr.-Nr.:~\@AutorMatrNr\\[0.65cm]
%
Betreuer/Themensteller:		& \@Themenstellername\\[0.05cm]
							& \@ThemenstellerZusatz\\[0.65cm]
%
Beginn:						& \@Beginn\\[0.05cm]
Eingereicht am:				& \@Ende\ in\ \@Ort
\end{tabular}
\vspace{1.9cm}%
\end{otherlanguage}%
\end{center}
%\newcommand{\StandAngabe}{\texttt{Übersetzt am: \today/\now}}
%\begin{center}
%\ifdraftclsmode\textcolor{red}{\StandAngabe}\fi
%\ifreview\textcolor[rgb]{0.05,0.54,0}{\StandAngabe}\fi
%\end{center}
\if@entwurf%
\vfill
\begin{center}
   \CmpTime
\end{center}
\vfill
\fi
\newpage
\cleardoublepage
}

\newcommand*{\Unterschriftsfeld}[1]{%
	\begin{tabular}{@{}p{0.35\textwidth}}
		\hline\\[-2ex]
		#1%
	\end{tabular}%
}

\newcommand{\DiplomErklaerung}{\newpage
\begin{otherlanguage}{ngerman}%
\chapter*{Eidesstattliche Erklärung}%\vspace{0,5cm}\\

Ich erkläre hiermit eidesstattlich, dass ich die vorgelegte Arbeit selbständig angefertigt habe. Die aus fremden Quellen direkt oder indirekt übernommenen Gedanken sind als solche kenntlich gemacht.\par
Die Arbeit wurde bisher keiner anderen Prüfungsbehörde vorgelegt.\vspace{4ex}\par
\@Ort, den \@Ende\vspace{10ex}\par
%\@Autorname
\Unterschriftsfeld{\@Autorname}
%\newpage\mbox{}\thispagestyle{empty}
%\newpage
\end{otherlanguage}%
\cleardoublepage

}

\newcommand{\Danksagung}{\newpage
	\begin{otherlanguage}{ngerman}%
		\chapter*{Danksagung}%\vspace{0,5cm}\\
		An dieser Stelle möchte ich mich bei meinem Betreuer Daniel Vietze, vom Lehrstuhl für Maschinenelemente an der Technischen Universität München, bedanken. Ohne seine Bereitschaft und kontinuierlichen Unterstützung während des gesamten Prozesses wäre diese Arbeit nicht entstanden.
	\end{otherlanguage}%
\cleardoublepage
}

\newcommand{\Aufgabenstellung}{\newpage
% \chapter*{Aufgabenstellung}%\\[0,1cm]
%\vspace*{-2.5ex}
% \vspace*{2ex}
\begin{otherlanguage}{ngerman}%
\begin{center}
	\bfseries\fsXVIseries
	\vspace*{0.2cm}%
	Aufgabenstellung der \@Arbeit\\
	\vspace{0.6cm}%
	\normalsize
	für \ifthenelse{\equal{\@AutorAnrede}{Herr}}{Herrn}{\@AutorAnrede} \@Autorname\\
\end{center}
\vspace{0.5cm}%
\textbf{Thema:}\\[0.1cm]
\end{otherlanguage}%
\ThemaPrimaersprache\\[0.3cm]
\begin{otherlanguage}{ngerman}%
\textbf{Aufgabenstellung:}\\[0.1cm]
\end{otherlanguage}%
% should be edited/corrected
%\input{Aufgabenstellung}
\InputIfFileExists{\@Aufgabepfad}{}{\ClassWarningNoLine{FZGdasa}{Das Pr\"aamble \protect\Aufgabepfad\space ist falsch oder nicht eingegeben. Definiere den Pfad der Aufgabenstellung mit \protect\Aufgabepfad{<Pfad>}}%
\textcolor{red}{\textit{Die Aufgabenstellung muss in einer TeX Datei (.tex) zusammengefasst werden. Das Präamble \texttt{$\backslash$Aufgabepfad\{<Pfad>\}} bindet den Inhalt ein.\newline HINWEIS: Die Aufgabenstellung hat eine Länge von ein bis eineinhalb Seiten.}}}

\begin{otherlanguage}{ngerman}%
{\itshape Die Arbeit wird nach ihrer erfolgreichen Fertigstellung in der Lehrstuhlbibliothek veröffentlicht. Informationen, Daten und EDV-Programme, die dem Studenten während der Bearbeitung dieser Arbeit vom betreuenden Assistenten oder anderen Mitarbeitern des Lehrstuhls zugänglich gemacht werden, sind als streng vertraulich zu behandeln und verbleiben Eigentum des Lehrstuhls. Die Nutzung ist ausschließlich für die Bearbeitung und Erstellung dieser Studienarbeit gestattet.

Die Nutzungsrechte an dieser Arbeit gehen an den Lehrstuhl für Maschinenelemente, Forschungsstelle für Zahnräder und Getriebesysteme, der Technischen Universität München über. Dem Verfasser wird ein privates Nutzungsrecht gewährt.}\vspace{4ex}\par
\@Ort, den \@Beginn\vspace{10ex}\par
%i.A.\vspace{1cm}\\
%\@Betreuername
\Unterschriftsfeld{\@Autorname}
\hfill
\Unterschriftsfeld{\@Betreuername}
%\newpage
\end{otherlanguage}%
\cleardoublepage
}

\renewcommand*\maketitle{%
\pagenumbering{Roman}
%
\titlepage
\Aufgabenstellung
\DiplomErklaerung
\Danksagung
\tableofcontents
%
% Abbildungsverzeichnis % Todo: Eventuell Option hinzufügen
%\cleardoublepage
%\listoffigures
%
% Tabellenverzeichnis % Todo: Eventuell Option hinzufügen
%\cleardoublepage
%\listoffigures
%\listoftables
%
\cleardoublepage
\pagenumbering{arabic}
}
\AtBeginDocument{\maketitle}

%\bibliographystyle{plaindin}
%-------------------------

