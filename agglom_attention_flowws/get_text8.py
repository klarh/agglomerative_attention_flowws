import argparse
import lzma
import os
import subprocess
import tempfile

import numpy as np

parser = argparse.ArgumentParser(
    description='Generate text8 data for embedding')
parser.add_argument('-o', '--output', default='text8.npz',
    help='Output filename for compressed text8 data')
parser.add_argument('--enwik9-location',
    help='Location of pre-downloaded enwik9.zip')

PERL_CONTENTS = r"""
#!/usr/bin/perl

# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
# letters (a-z, converted from A-Z), and spaces (never consecutive).
# All other characters are converted to spaces.  Only text which normally appears
# in the web browser is displayed.  Tables are removed.  Image captions are
# preserved.  Links are converted to normal text.  Digits are spelled out.

# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.

$/=">";                     # input record separator
while (<>) {
  if (/<text /) {$text=1;}  # remove all but between <text> ... </text>
  if (/#redirect/i) {$text=0;}  # remove #REDIRECT
  if ($text) {

    # Remove any text not normally visible
    if (/<\/text>/) {$text=0;}
    s/<.*>//;               # remove xml tags
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    s/\{\{[^}]*\}\}//g;         # remove {{icons}} and {tables}
    s/\{[^}]*\}//g;
    s/\[//g;                # remove [ and ]
    s/\]//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars

    # convert to lowercase letters and spaces, spell digits
    $_=" $_ ";
    tr/A-Z/a-z/;
    s/0/ zero /g;
    s/1/ one /g;
    s/2/ two /g;
    s/3/ three /g;
    s/4/ four /g;
    s/5/ five /g;
    s/6/ six /g;
    s/7/ seven /g;
    s/8/ eight /g;
    s/9/ nine /g;
    tr/a-z/ /cs;
    chop;
    print $_;
  }
}
"""

def main():
    args = parser.parse_args()

    target_fname = os.path.abspath(args.output)

    if args.enwik9_location:
        enwik9 = os.path.abspath(args.enwik9_location)
    else:
        enwik9 = 'enwik9.zip'

    with tempfile.TemporaryDirectory() as path:
        os.chdir(path)

        with open('text8_conversion.pl', 'w') as f:
            f.write(PERL_CONTENTS)

        if not args.enwik9_location:
            command = ['wget', 'http://mattmahoney.net/dc/enwik9.zip']
            subprocess.check_call(command)

        command = ('unzip -p {} enwik9 | '
                   'perl text8_conversion.pl | '
                   'head -c 100000000 > text8').format(enwik9)
        subprocess.check_call(command, shell=True)

        text8 = np.memmap('text8', mode='r')
        np.savez_compressed(target_fname, text8=text8)

if __name__ == '__main__': main()
