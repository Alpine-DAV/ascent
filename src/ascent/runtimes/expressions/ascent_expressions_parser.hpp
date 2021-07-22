/* A Bison parser, made by GNU Bison 3.6.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_ASCENT_ASCENT_EXPRESSIONS_PARSER_HPP_INCLUDED
# define YY_ASCENT_ASCENT_EXPRESSIONS_PARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int ascentdebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    TIDENTIFIER = 258,             /* TIDENTIFIER  */
    TINTEGER = 259,                /* TINTEGER  */
    TDOUBLE = 260,                 /* TDOUBLE  */
    TSTRING = 261,                 /* TSTRING  */
    TOR = 262,                     /* TOR  */
    TAND = 263,                    /* TAND  */
    TNOT = 264,                    /* TNOT  */
    TTRUE = 265,                   /* TTRUE  */
    TFALSE = 266,                  /* TFALSE  */
    TIF = 267,                     /* TIF  */
    TTHEN = 268,                   /* TTHEN  */
    TELSE = 269,                   /* TELSE  */
    TAEQ = 270,                    /* TAEQ  */
    TCEQ = 271,                    /* TCEQ  */
    TCNE = 272,                    /* TCNE  */
    TCLT = 273,                    /* TCLT  */
    TCLE = 274,                    /* TCLE  */
    TCGT = 275,                    /* TCGT  */
    TCGE = 276,                    /* TCGE  */
    TLPAREN = 277,                 /* TLPAREN  */
    TRPAREN = 278,                 /* TRPAREN  */
    TLBRACKET = 279,               /* TLBRACKET  */
    TRBRACKET = 280,               /* TRBRACKET  */
    TCOMMA = 281,                  /* TCOMMA  */
    TDOT = 282,                    /* TDOT  */
    TENDEXPR = 283,                /* TENDEXPR  */
    TPLUS = 284,                   /* TPLUS  */
    TMINUS = 285,                  /* TMINUS  */
    TMUL = 286,                    /* TMUL  */
    TDIV = 287,                    /* TDIV  */
    TMOD = 288,                    /* TMOD  */
    TNEG = 289                     /* TNEG  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 21 "parser.y"

 ASTNode                     *node;
 ASTBlock                    *block;
 ASTExpression               *expr;
 ASTIdentifier               *ident;
 ASTString                   *string_literal;
 ASTBoolean                  *bool_literal;
 ASTExpressionList           *expr_list;
 ASTNamedExpression          *named_expr;
 ASTNamedExpressionList      *named_expr_list;
 ASTArguments                *args;
 std::string                 *string;
 int token;

#line 113 "ascent_expressions_parser.hpp"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE ascentlval;

int ascentparse (void);
/* "%code provides" blocks.  */
#line 16 "parser.y"

  ASTNode *get_result();

#line 130 "ascent_expressions_parser.hpp"

#endif /* !YY_ASCENT_ASCENT_EXPRESSIONS_PARSER_HPP_INCLUDED  */
