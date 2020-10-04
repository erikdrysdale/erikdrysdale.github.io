"""
REPLACE ALL $...$ WITH \\(...\\)
"""

import os
import pandas as pd
from datetime import datetime

dir_base = os.getcwd()
dir_posts = os.path.join(dir_base, '..', '..', '_posts')
dir_output = os.path.join(dir_base, 'output')
if not os.path.exists(dir_output):
    os.mkdir(dir_output)

fn_posts = pd.Series(os.listdir(dir_posts))
fn_posts = fn_posts[fn_posts.str.contains('^[0-9]')]
date_posts = pd.to_datetime(fn_posts.str.split('\\-[A-Za-z]',1,True).iloc[:,0])
yy_mm = pd.to_datetime('2020-09-01') #str(datetime.today().year)+'-'+str(datetime.today().month)+'-01'
fn_posts = fn_posts[date_posts >= yy_mm].to_list()

for ii, fn in enumerate(fn_posts):
    print('Post: %s (%i of %i)' % (fn, ii+1, len(fn_posts)))
    connection = open(os.path.join(dir_posts, fn), 'r')
    lines = pd.Series(connection.readlines())
    connection.close()
    eq_labels = lines[lines.str.contains('\\\\label\\{eq\\:')]
    if len(eq_labels) > 0:
        print('Has label/eqref')
        tick = 0
        for kk, eq in eq_labels.iteritems():
            if eq.split('\\label')[0][-1] == ' ':
                print('Needs tag')
                tick += 1
                rep = '\\tag{'+str(tick)+'}\\label'
                lines[kk] = eq.replace('\\label', rep)
            else:
                print('Already has tag')
    # Remove double $ dollars temporarily
    lines = lines.str.replace('\\${2}', '@@')
    # Replace currency dollar signs
    lines = lines.str.replace('\\\\\\$', '~~')

    should_skip = False
    for jj, line in enumerate(lines):
        # ignore until we find
        if ('{%' in line) | ('blockquote' in line) | ('```' in line):
            should_skip = not should_skip
            #print('skipping: %s, line: %s' % (should_skip,line))
        if should_skip:
            continue
        if '$' in line:
            ss = line.split('$')
            n = len(ss)
            assert n % 2 == 1
            assert n >= 3
            tmp = ss[0]
            for idx in range(1, n):
                if idx % 2 == 1:
                    tmp += '\\\\(' + ss[idx]
                else:
                    tmp += '\\\\)' + ss[idx]
            print('--------Replacement for: %i-----------' % jj)
            print('Original: %s\nNew: %s' % (line, tmp))
            lines[jj] = tmp
    # Under the blocks
    lines = lines.str.replace('@@','$$')
    lines = lines.str.replace('~~', '\\\\$')
    # Now write
    copy = open(os.path.join(dir_output, fn), 'w')
    for line in lines:
        copy.write(line)
    copy.close()
