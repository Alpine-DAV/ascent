//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/* A Bison parser, made by GNU Bison 3.6.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.6"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1


/* Substitute the variable and function names.  */
#define yyparse         ascentparse
#define yylex           ascentlex
#define yyerror         ascenterror
#define yydebug         ascentdebug
#define yynerrs         ascentnerrs
#define yylval          ascentlval
#define yychar          ascentchar

/* First part of user prologue.  */
#line 1 "parser.y"

  #include "ascent_expressions_ast.hpp"
  #include <cstdio>
  #include <cstdlib>

  ASTNode *root_node; /* the top level root node of our final AST */

  extern int yylex();
  extern void scan_string(const char *);
  void yyerror(const char *s)
  {
    throw "syntax error";
  }
  ASTNode *get_result() { return root_node; }

#line 94 "ascent_expressions_parser.cpp"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
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

#line 193 "ascent_expressions_parser.cpp"

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

#line 210 "ascent_expressions_parser.cpp"

#endif /* !YY_ASCENT_ASCENT_EXPRESSIONS_PARSER_HPP_INCLUDED  */
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_TIDENTIFIER = 3,                /* TIDENTIFIER  */
  YYSYMBOL_TINTEGER = 4,                   /* TINTEGER  */
  YYSYMBOL_TDOUBLE = 5,                    /* TDOUBLE  */
  YYSYMBOL_TSTRING = 6,                    /* TSTRING  */
  YYSYMBOL_TOR = 7,                        /* TOR  */
  YYSYMBOL_TAND = 8,                       /* TAND  */
  YYSYMBOL_TNOT = 9,                       /* TNOT  */
  YYSYMBOL_TTRUE = 10,                     /* TTRUE  */
  YYSYMBOL_TFALSE = 11,                    /* TFALSE  */
  YYSYMBOL_TIF = 12,                       /* TIF  */
  YYSYMBOL_TTHEN = 13,                     /* TTHEN  */
  YYSYMBOL_TELSE = 14,                     /* TELSE  */
  YYSYMBOL_TAEQ = 15,                      /* TAEQ  */
  YYSYMBOL_TCEQ = 16,                      /* TCEQ  */
  YYSYMBOL_TCNE = 17,                      /* TCNE  */
  YYSYMBOL_TCLT = 18,                      /* TCLT  */
  YYSYMBOL_TCLE = 19,                      /* TCLE  */
  YYSYMBOL_TCGT = 20,                      /* TCGT  */
  YYSYMBOL_TCGE = 21,                      /* TCGE  */
  YYSYMBOL_TLPAREN = 22,                   /* TLPAREN  */
  YYSYMBOL_TRPAREN = 23,                   /* TRPAREN  */
  YYSYMBOL_TLBRACKET = 24,                 /* TLBRACKET  */
  YYSYMBOL_TRBRACKET = 25,                 /* TRBRACKET  */
  YYSYMBOL_TCOMMA = 26,                    /* TCOMMA  */
  YYSYMBOL_TDOT = 27,                      /* TDOT  */
  YYSYMBOL_TENDEXPR = 28,                  /* TENDEXPR  */
  YYSYMBOL_TPLUS = 29,                     /* TPLUS  */
  YYSYMBOL_TMINUS = 30,                    /* TMINUS  */
  YYSYMBOL_TMUL = 31,                      /* TMUL  */
  YYSYMBOL_TDIV = 32,                      /* TDIV  */
  YYSYMBOL_TMOD = 33,                      /* TMOD  */
  YYSYMBOL_TNEG = 34,                      /* TNEG  */
  YYSYMBOL_YYACCEPT = 35,                  /* $accept  */
  YYSYMBOL_root = 36,                      /* root  */
  YYSYMBOL_block = 37,                     /* block  */
  YYSYMBOL_stmts = 38,                     /* stmts  */
  YYSYMBOL_stmt = 39,                      /* stmt  */
  YYSYMBOL_ident = 40,                     /* ident  */
  YYSYMBOL_numeric = 41,                   /* numeric  */
  YYSYMBOL_call_args = 42,                 /* call_args  */
  YYSYMBOL_pos_args = 43,                  /* pos_args  */
  YYSYMBOL_named_args = 44,                /* named_args  */
  YYSYMBOL_named_arg = 45,                 /* named_arg  */
  YYSYMBOL_if_expr = 46,                   /* if_expr  */
  YYSYMBOL_string_literal = 47,            /* string_literal  */
  YYSYMBOL_array_access = 48,              /* array_access  */
  YYSYMBOL_dot_access = 49,                /* dot_access  */
  YYSYMBOL_list = 50,                      /* list  */
  YYSYMBOL_bool_literal = 51,              /* bool_literal  */
  YYSYMBOL_expr = 52                       /* expr  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   306

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  35
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  18
/* YYNRULES -- Number of rules.  */
#define YYNRULES  52
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  91

#define YYMAXUTOK   289


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint8 yyrline[] =
{
       0,    86,    86,    89,    90,    93,    94,    97,   100,   103,
     104,   107,   108,   109,   110,   113,   114,   117,   118,   121,
     124,   127,   130,   133,   136,   137,   140,   141,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   154,   155,
     156,   157,   158,   159,   160,   161,   162,   163,   164,   165,
     166,   167,   168
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "TIDENTIFIER",
  "TINTEGER", "TDOUBLE", "TSTRING", "TOR", "TAND", "TNOT", "TTRUE",
  "TFALSE", "TIF", "TTHEN", "TELSE", "TAEQ", "TCEQ", "TCNE", "TCLT",
  "TCLE", "TCGT", "TCGE", "TLPAREN", "TRPAREN", "TLBRACKET", "TRBRACKET",
  "TCOMMA", "TDOT", "TENDEXPR", "TPLUS", "TMINUS", "TMUL", "TDIV", "TMOD",
  "TNEG", "$accept", "root", "block", "stmts", "stmt", "ident", "numeric",
  "call_args", "pos_args", "named_args", "named_arg", "if_expr",
  "string_literal", "array_access", "dot_access", "list", "bool_literal",
  "expr", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289
};
#endif

#define YYPACT_NINF (-53)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     -53,     3,   -53,    83,   -53,   -53,   -53,   -53,   -53,    83,
     -53,   -53,    83,    83,    50,    83,   -53,   -14,   -53,   -53,
     -53,   -53,   -53,   -53,   -53,   101,   -18,   263,   128,   146,
     -53,    -8,   227,   -22,    83,    83,    83,    83,    83,    83,
      83,    83,    83,    83,    83,     4,   -53,    83,    83,    83,
      83,    83,    83,   -53,   -53,    83,   164,    35,    -2,    -6,
       6,   -53,   245,   263,    -5,    -5,   273,   273,   273,   273,
     182,   -53,    79,    79,   -22,   -22,   -22,   209,   227,   -53,
      83,   -53,    83,    20,   -53,    83,   227,     6,    14,   -53,
     -22
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       5,     0,     2,     0,     1,     8,     9,    10,    21,     0,
      26,    27,     0,     0,     0,     0,     6,    29,    36,    34,
      31,    32,    33,    35,    30,     4,    29,    51,     0,     0,
      24,     0,    15,    42,     0,    11,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     3,     0,     0,     0,
       0,     0,     0,    52,    25,     0,     0,    29,     0,    12,
      13,    17,    49,    50,    43,    44,    45,    46,    47,    48,
       0,    23,    40,    41,    37,    38,    39,     0,    16,     7,
       0,    28,     0,     0,    22,     0,    19,    14,     0,    18,
      20
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -53,   -53,   -53,   -53,   -53,     8,   -53,   -53,     7,   -52,
     -32,   -53,   -53,   -53,   -53,   -53,   -53,    -3
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     1,     2,     3,    16,    26,    18,    58,    31,    60,
      61,    19,    20,    21,    22,    23,    24,    32
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
      25,    34,    44,     4,    35,    45,    27,    71,    35,    28,
      29,    17,    33,    40,    41,    42,    43,    54,    55,    44,
      82,    81,    45,     5,    47,    48,    49,    50,    51,    80,
      87,    56,    83,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    59,    57,    72,    73,    74,    75,    76,    77,
      80,    89,    78,     5,     6,     7,     8,    35,     0,     9,
      10,    11,    12,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    13,     0,    14,    30,     0,    86,     0,    78,
      15,     0,    90,     0,     0,     0,     5,     6,     7,     8,
      57,    88,     9,    10,    11,    12,     0,     0,     0,     0,
       0,     0,     0,    44,     0,    13,    45,    14,    36,    37,
      49,    50,    51,    15,     0,     0,     0,    38,    39,    40,
      41,    42,    43,     0,     0,    44,     0,     0,    45,    46,
      47,    48,    49,    50,    51,    36,    37,     0,     0,     0,
       0,    52,     0,     0,    38,    39,    40,    41,    42,    43,
       0,     0,    44,    36,    37,    45,     0,    47,    48,    49,
      50,    51,    38,    39,    40,    41,    42,    43,     0,    53,
      44,    36,    37,    45,     0,    47,    48,    49,    50,    51,
      38,    39,    40,    41,    42,    43,     0,     0,    44,    36,
      37,    45,    79,    47,    48,    49,    50,    51,    38,    39,
      40,    41,    42,    43,     0,     0,    44,    84,     0,    45,
       0,    47,    48,    49,    50,    51,    36,    37,     0,     0,
       0,     0,     0,    85,     0,    38,    39,    40,    41,    42,
      43,     0,     0,    44,    36,    37,    45,     0,    47,    48,
      49,    50,    51,    38,    39,    40,    41,    42,    43,     0,
       0,    44,     0,    37,    45,     0,    47,    48,    49,    50,
      51,    38,    39,    40,    41,    42,    43,     0,     0,    44,
       0,     0,    45,     0,    47,    48,    49,    50,    51,    38,
      39,    40,    41,    42,    43,     0,     0,    44,     0,     0,
      45,     0,    47,    48,    49,    50,    51,    44,     0,     0,
      45,     0,    47,    48,    49,    50,    51
};

static const yytype_int8 yycheck[] =
{
       3,    15,    24,     0,    22,    27,     9,     3,    22,    12,
      13,     3,    15,    18,    19,    20,    21,    25,    26,    24,
      26,    23,    27,     3,    29,    30,    31,    32,    33,    15,
      82,    34,    26,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    35,    35,    47,    48,    49,    50,    51,    52,
      15,    83,    55,     3,     4,     5,     6,    22,    -1,     9,
      10,    11,    12,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    22,    -1,    24,    25,    -1,    80,    -1,    82,
      30,    -1,    85,    -1,    -1,    -1,     3,     4,     5,     6,
      82,    83,     9,    10,    11,    12,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    24,    -1,    22,    27,    24,     7,     8,
      31,    32,    33,    30,    -1,    -1,    -1,    16,    17,    18,
      19,    20,    21,    -1,    -1,    24,    -1,    -1,    27,    28,
      29,    30,    31,    32,    33,     7,     8,    -1,    -1,    -1,
      -1,    13,    -1,    -1,    16,    17,    18,    19,    20,    21,
      -1,    -1,    24,     7,     8,    27,    -1,    29,    30,    31,
      32,    33,    16,    17,    18,    19,    20,    21,    -1,    23,
      24,     7,     8,    27,    -1,    29,    30,    31,    32,    33,
      16,    17,    18,    19,    20,    21,    -1,    -1,    24,     7,
       8,    27,    28,    29,    30,    31,    32,    33,    16,    17,
      18,    19,    20,    21,    -1,    -1,    24,    25,    -1,    27,
      -1,    29,    30,    31,    32,    33,     7,     8,    -1,    -1,
      -1,    -1,    -1,    14,    -1,    16,    17,    18,    19,    20,
      21,    -1,    -1,    24,     7,     8,    27,    -1,    29,    30,
      31,    32,    33,    16,    17,    18,    19,    20,    21,    -1,
      -1,    24,    -1,     8,    27,    -1,    29,    30,    31,    32,
      33,    16,    17,    18,    19,    20,    21,    -1,    -1,    24,
      -1,    -1,    27,    -1,    29,    30,    31,    32,    33,    16,
      17,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
      27,    -1,    29,    30,    31,    32,    33,    24,    -1,    -1,
      27,    -1,    29,    30,    31,    32,    33
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    36,    37,    38,     0,     3,     4,     5,     6,     9,
      10,    11,    12,    22,    24,    30,    39,    40,    41,    46,
      47,    48,    49,    50,    51,    52,    40,    52,    52,    52,
      25,    43,    52,    52,    15,    22,     7,     8,    16,    17,
      18,    19,    20,    21,    24,    27,    28,    29,    30,    31,
      32,    33,    13,    23,    25,    26,    52,    40,    42,    43,
      44,    45,    52,    52,    52,    52,    52,    52,    52,    52,
      52,     3,    52,    52,    52,    52,    52,    52,    52,    28,
      15,    23,    26,    26,    25,    14,    52,    44,    40,    45,
      52
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
       0,    35,    36,    37,    37,    38,    38,    39,    40,    41,
      41,    42,    42,    42,    42,    43,    43,    44,    44,    45,
      46,    47,    48,    49,    50,    50,    51,    51,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     3,     2,     0,     2,     4,     1,     1,
       1,     0,     1,     1,     3,     1,     3,     1,     3,     3,
       6,     1,     4,     3,     2,     3,     1,     1,     4,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       3,     3,     2,     3,     3,     3,     3,     3,     3,     3,
       3,     2,     3
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
# ifndef YY_LOCATION_PRINT
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yykind < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yykind], *yyvaluep);
# endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize;

    /* The state stack.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss;
    yy_state_t *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yynerrs = 0;
  yystate = 0;
  yyerrstatus = 0;

  yystacksize = YYINITDEPTH;
  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;


  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2:
#line 86 "parser.y"
             { root_node = (yyvsp[0].block); }
#line 1338 "ascent_expressions_parser.cpp"
    break;

  case 3:
#line 89 "parser.y"
                            { (yyval.block) = new ASTBlock((yyvsp[-2].named_expr_list), (yyvsp[-1].expr)); }
#line 1344 "ascent_expressions_parser.cpp"
    break;

  case 4:
#line 90 "parser.y"
               { (yyval.block) = new ASTBlock((yyvsp[-1].named_expr_list), (yyvsp[0].expr)); }
#line 1350 "ascent_expressions_parser.cpp"
    break;

  case 5:
#line 93 "parser.y"
        { (yyval.named_expr_list) = new ASTNamedExpressionList(); }
#line 1356 "ascent_expressions_parser.cpp"
    break;

  case 6:
#line 94 "parser.y"
               { (yyvsp[-1].named_expr_list)->push_back((yyvsp[0].named_expr)); }
#line 1362 "ascent_expressions_parser.cpp"
    break;

  case 7:
#line 97 "parser.y"
                                { (yyval.named_expr) = new ASTNamedExpression((yyvsp[-3].ident), (yyvsp[-1].expr)); }
#line 1368 "ascent_expressions_parser.cpp"
    break;

  case 8:
#line 100 "parser.y"
                    { (yyval.ident) = new ASTIdentifier(*(yyvsp[0].string)); delete (yyvsp[0].string); }
#line 1374 "ascent_expressions_parser.cpp"
    break;

  case 9:
#line 103 "parser.y"
                   { (yyval.expr) = new ASTInteger(std::stoi(*(yyvsp[0].string))); delete (yyvsp[0].string); }
#line 1380 "ascent_expressions_parser.cpp"
    break;

  case 10:
#line 104 "parser.y"
            { (yyval.expr) = new ASTDouble(std::stod(*(yyvsp[0].string))); delete (yyvsp[0].string); }
#line 1386 "ascent_expressions_parser.cpp"
    break;

  case 11:
#line 107 "parser.y"
                       { (yyval.args) = new ASTArguments(nullptr, nullptr); }
#line 1392 "ascent_expressions_parser.cpp"
    break;

  case 12:
#line 108 "parser.y"
             { (yyval.args) = new ASTArguments((yyvsp[0].expr_list), nullptr); }
#line 1398 "ascent_expressions_parser.cpp"
    break;

  case 13:
#line 109 "parser.y"
               { (yyval.args) = new ASTArguments(nullptr, (yyvsp[0].named_expr_list)); }
#line 1404 "ascent_expressions_parser.cpp"
    break;

  case 14:
#line 110 "parser.y"
                               { (yyval.args) = new ASTArguments((yyvsp[-2].expr_list), (yyvsp[0].named_expr_list)); }
#line 1410 "ascent_expressions_parser.cpp"
    break;

  case 15:
#line 113 "parser.y"
                { (yyval.expr_list) = new ASTExpressionList(); (yyval.expr_list)->exprs.push_back((yyvsp[0].expr)); }
#line 1416 "ascent_expressions_parser.cpp"
    break;

  case 16:
#line 114 "parser.y"
                         { (yyvsp[-2].expr_list)->exprs.push_back((yyvsp[0].expr)); }
#line 1422 "ascent_expressions_parser.cpp"
    break;

  case 17:
#line 117 "parser.y"
                       { (yyval.named_expr_list) = new ASTNamedExpressionList(); (yyval.named_expr_list)->push_back((yyvsp[0].named_expr)); }
#line 1428 "ascent_expressions_parser.cpp"
    break;

  case 18:
#line 118 "parser.y"
                                { (yyvsp[-2].named_expr_list)->push_back((yyvsp[0].named_expr)); }
#line 1434 "ascent_expressions_parser.cpp"
    break;

  case 19:
#line 121 "parser.y"
                            { (yyval.named_expr) = new ASTNamedExpression((yyvsp[-2].ident), (yyvsp[0].expr)); }
#line 1440 "ascent_expressions_parser.cpp"
    break;

  case 20:
#line 124 "parser.y"
                                        { (yyval.expr) = new ASTIfExpr((yyvsp[-4].expr), (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 1446 "ascent_expressions_parser.cpp"
    break;

  case 21:
#line 127 "parser.y"
                         { (yyval.string_literal) = new ASTString(*(yyvsp[0].string)); delete (yyvsp[0].string); }
#line 1452 "ascent_expressions_parser.cpp"
    break;

  case 22:
#line 130 "parser.y"
                                             { (yyval.expr) = new ASTArrayAccess((yyvsp[-3].expr), (yyvsp[-1].expr)); }
#line 1458 "ascent_expressions_parser.cpp"
    break;

  case 23:
#line 133 "parser.y"
                                   { (yyval.expr) = new ASTDotAccess((yyvsp[-2].expr), *(yyvsp[0].string)); delete (yyvsp[0].string); }
#line 1464 "ascent_expressions_parser.cpp"
    break;

  case 24:
#line 136 "parser.y"
                           {(yyval.expr_list) = new ASTExpressionList();}
#line 1470 "ascent_expressions_parser.cpp"
    break;

  case 25:
#line 137 "parser.y"
                                    { (yyval.expr_list) = (yyvsp[-1].expr_list); }
#line 1476 "ascent_expressions_parser.cpp"
    break;

  case 26:
#line 140 "parser.y"
                     { (yyval.bool_literal) = new ASTBoolean((yyvsp[0].token)); }
#line 1482 "ascent_expressions_parser.cpp"
    break;

  case 27:
#line 141 "parser.y"
             { (yyval.bool_literal) = new ASTBoolean((yyvsp[0].token)); }
#line 1488 "ascent_expressions_parser.cpp"
    break;

  case 28:
#line 144 "parser.y"
                                       { (yyval.expr) = new ASTMethodCall((yyvsp[-3].ident), (yyvsp[-1].args)); }
#line 1494 "ascent_expressions_parser.cpp"
    break;

  case 29:
#line 145 "parser.y"
          { (yyval.expr) = (yyvsp[0].ident); }
#line 1500 "ascent_expressions_parser.cpp"
    break;

  case 30:
#line 146 "parser.y"
                 { (yyval.expr) = (yyvsp[0].bool_literal); }
#line 1506 "ascent_expressions_parser.cpp"
    break;

  case 31:
#line 147 "parser.y"
                   { (yyval.expr) = (yyvsp[0].string_literal); }
#line 1512 "ascent_expressions_parser.cpp"
    break;

  case 32:
#line 148 "parser.y"
                 { (yyval.expr) = (yyvsp[0].expr); }
#line 1518 "ascent_expressions_parser.cpp"
    break;

  case 33:
#line 149 "parser.y"
               { (yyval.expr) = (yyvsp[0].expr); }
#line 1524 "ascent_expressions_parser.cpp"
    break;

  case 34:
#line 150 "parser.y"
            { (yyval.expr) = (yyvsp[0].expr); }
#line 1530 "ascent_expressions_parser.cpp"
    break;

  case 35:
#line 151 "parser.y"
         { (yyval.expr) = (yyvsp[0].expr_list); }
#line 1536 "ascent_expressions_parser.cpp"
    break;

  case 37:
#line 153 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1542 "ascent_expressions_parser.cpp"
    break;

  case 38:
#line 154 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1548 "ascent_expressions_parser.cpp"
    break;

  case 39:
#line 155 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1554 "ascent_expressions_parser.cpp"
    break;

  case 40:
#line 156 "parser.y"
                    { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1560 "ascent_expressions_parser.cpp"
    break;

  case 41:
#line 157 "parser.y"
                     { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1566 "ascent_expressions_parser.cpp"
    break;

  case 42:
#line 158 "parser.y"
                           { (yyval.expr) = new ASTBinaryOp(new ASTInteger(-1),TMUL,(yyvsp[0].expr)); }
#line 1572 "ascent_expressions_parser.cpp"
    break;

  case 43:
#line 159 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1578 "ascent_expressions_parser.cpp"
    break;

  case 44:
#line 160 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1584 "ascent_expressions_parser.cpp"
    break;

  case 45:
#line 161 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1590 "ascent_expressions_parser.cpp"
    break;

  case 46:
#line 162 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1596 "ascent_expressions_parser.cpp"
    break;

  case 47:
#line 163 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1602 "ascent_expressions_parser.cpp"
    break;

  case 48:
#line 164 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1608 "ascent_expressions_parser.cpp"
    break;

  case 49:
#line 165 "parser.y"
                  { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1614 "ascent_expressions_parser.cpp"
    break;

  case 50:
#line 166 "parser.y"
                   { (yyval.expr) = new ASTBinaryOp((yyvsp[-2].expr), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1620 "ascent_expressions_parser.cpp"
    break;

  case 51:
#line 167 "parser.y"
              { (yyval.expr) = new ASTBinaryOp(new ASTExpression(), (yyvsp[-1].token), (yyvsp[0].expr)); }
#line 1626 "ascent_expressions_parser.cpp"
    break;

  case 52:
#line 168 "parser.y"
                         { (yyval.expr) = (yyvsp[-1].expr); }
#line 1632 "ascent_expressions_parser.cpp"
    break;


#line 1636 "ascent_expressions_parser.cpp"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  // Pop stack until we find a state that shifts the error token.
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if !defined yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif


/*-----------------------------------------------------.
| yyreturn -- parsing is finished, return the result.  |
`-----------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 171 "parser.y"

