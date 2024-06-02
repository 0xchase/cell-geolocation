%% 
%% This is file `paper.cls',
%% by Dave Levin <dml@cs.umd.edu>
%%
%% Are you fed up with having to come up with new paper formatting every
%% time you submit to a new conference? Well look no further; this is a
%% unified conference format file to be used with a whole slew of
%% conferences in security and networking!
%%
%% Simply specify which conference you want as the sole option to your
%% \documentclass command. For example:
%%
%%      \documentclass[usenix]{paper}
%%                     ^^^^^^ just change this
%%
%% Use the provided paper.tex and fill in the commands and you're set!
%%
%% Currently supports:
%% ccs imc ndss oakland sigcomm tma usenix www
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{paper}[dml unified paper formatting as of Jan 2024]

\newif\ifabstractfirst
\newif\ifieeeformat

\newif\ifccs
\newif\ifimc
\newif\ifndss
\newif\ifoakland
\newif\ifsigcomm
\newif\iftma
\newif\ifusenix
\newif\ifwww

\DeclareOption{ccs}{\ccstrue \abstractfirsttrue \ieeeformatfalse}
\DeclareOption{imc}{\imctrue \abstractfirsttrue \ieeeformatfalse}
\DeclareOption{ndss}{\ndsstrue \abstractfirstfalse \ieeeformattrue}
\DeclareOption{oakland}{\oaklandtrue \abstractfirstfalse \ieeeformattrue}
\DeclareOption{sigcomm}{\sigcommtrue \abstractfirsttrue \ieeeformatfalse}
\DeclareOption{tma}{\tmatrue \abstractfirstfalse \ieeeformattrue}
\DeclareOption{usenix}{\usenixtrue \abstractfirstfalse \ieeeformatfalse}
\DeclareOption{test}{\PackageError{paper}{Test failed}{Failed nice}}
\DeclareOption{www}{\wwwtrue \abstractfirsttrue \ieeeformatfalse}

\ProcessOptions\relax

\usepackage{etoolbox}

\ifccs \else \ifimc \else \ifndss \else \ifoakland \else \ifsigcomm
\else \iftma \else \ifusenix \else \ifwww \else
	\PackageError{paper}{You must specify one of the following
	as an option to documentclass in paper.tex:
	ccs, imc, ndss, oakland, sigcomm, tma, usenix, www}
	{ Fix your documentclass line }
\fi \fi \fi \fi \fi \fi \fi \fi

\newcommand{\Title}[1]{\title{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CCS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifccs
\typeout{🏃 Loading CCS conference files}

\LoadClass[sigconf,anonymous]{templates/ccs/acmart}

\newcommand{\ConferenceYear}{2024\xspace}
\newcommand{\ConferenceLocation}{Salt Lake City, USA}
\newcommand{\ConferenceDate}{Due April 14, 2024}

\fancyhf{}
\fancyhead[C]{Anonymous submission \#\PaperNumber{} to ACM CCS \ConferenceYear{}}
\fancyfoot[C]{\thepage}

\setcopyright{none}
\acmConference[Anonymous Submission to ACM CCS \ConferenceYear] {ACM Conference on Computer and Communications Security} {\ConferenceDate}{\ConferenceLocation}
\acmYear{\ConferenceYear}

\settopmatter{printacmref=false, printccs=true, printfolios=true}
\bibliographystyle{templates/ccs/ACM-Reference-Format}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifimc
\typeout{🏃 Loading IMC conference files}

\LoadClass[10pt,sigconf,letterpaper,anonymous,nonacm]{templates/imc/acmart}

\newcommand{\ConferenceYear}{2024\xspace}
\newcommand{\ConferenceLocation}{Madrid, Spain}
\newcommand{\ConferenceDate}{November 2--4, 2024}

\setcopyright{none}
\acmConference[Anonymous Submission to ACM IMC \ConferenceYear] {ACM Internet Measurement Conference} {\ConferenceDate}{\ConferenceLocation}
\acmYear{\ConferenceYear}

\bibliographystyle{templates/imc/ACM-Reference-Format}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NDSS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifndss
\typeout{🏃 Loading NDSS conference files}

\LoadClass[conference]{templates/ndss/IEEEtran}
\pagestyle{plain}
\bibliographystyle{templates/ndss/IEEEtranS}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Oakland
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifoakland
\typeout{🏃 Loading Oakland conference files}

\LoadClass[conference,compsoc]{templates/oakland/IEEEtran}
\pagestyle{plain}
\bibliographystyle{templates/oakland/IEEEtranS}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGCOMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifsigcomm
\typeout{🏃 Loading SIGCOMM conference files}

\LoadClass[10pt,sigconf,letterpaper,anonymous,nonacm]{templates/sigcomm/acmart}
\bibliographystyle{templates/sigcomm/ACM-Reference-Format}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TMA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\iftma
\typeout{🏃 Loading TMA conference files}

\LoadClass[conference]{templates/tma/IEEEtran}
\bibliographystyle{templates/tma/IEEEtran}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USENIX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifusenix
\typeout{🏃 Loading USENIX conference files}

\LoadClass[letterpaper,twocolumn,10pt]{article}
\usepackage{templates/usenix/usenix}
\bibliographystyle{plain}
\renewcommand{\Title}[1]{\title{\Large \bf \PaperTitle}}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WWW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifwww
\typeout{🏃 Loading WWW conference files}

\PackageError{conference}{WWW loaded}{www loaded}
\LoadClass[sigconf,anonymous]{templates/www/acmart}
\bibliographystyle{templates/www/ACM-Reference-Format}

\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title, authors, abstract
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\Abstract}[1]{ %
	\Title{\PaperTitle} %
	\author{Paper \#\PaperNumber} %
	\ifabstractfirst %
		\begin{abstract} %
			\expandafter\input {#1} %
		\end{abstract} %
		\maketitle %
	\else %
		\maketitle %
		\begin{abstract} %
			\expandafter\input {#1} %
		\end{abstract} %
	\fi %
}


