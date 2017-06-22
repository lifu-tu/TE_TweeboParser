#!/usr/bin/perl
binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
if (scalar(@ARGV) != 1) {
    print STDERR "Usage: proc-tweets-uniformize-conll.pl field_of_word(zero-indexed) < in > out\n";
    exit;
}
$field = $ARGV[0];

while (<STDIN>) {
	$line = $_;
	chomp($line);
	# tokenize.
	if ($line =~ m/^\s*$/) {
		print $line."\n";
	} else {
		@toks = split /\t/, $line;
		$n = @toks;
		for ($i = 0; $i < $n; $i++) {
			if ($i > 0) {
				print "\t";
			}
			if ($i == $field) {
				if ($toks[$i] =~ m/^\@.+/) {
					print "<\@MENTION>";
        	        	} elsif ($toks[$i] =~ m/^\<URL-.+\>$/) {
                        		print $toks[$i];
				} elsif ($toks[$i] =~ m/^http\:\/\/(www\.)?([^\/]+)\/?/) {
					print "<URL-$2>";
				} elsif ($toks[$i] =~ m/^(www\.)?([^\/]+\.com)\/?/) {
					print "<URL-$2>";
				} elsif ($toks[$i] =~ m/^\<\@MENTION\>$/) {
					print $toks[$i];
				} else {
					print lc($toks[$i]);
				}
			} else {
				print $toks[$i];
			}
		}
		print "\n";
	}
}

