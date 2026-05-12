# How to build the website locally

## Prerequisites

Ruby 3.x+ from Homebrew is required. The system Ruby (2.6) is too old. Check with:

```bash
ruby --version
```

If it shows `/usr/bin/ruby` or version 2.6, install and activate the Homebrew version:

```bash
brew install ruby
```

Then add the following to your `~/.zshrc` (or `~/.bashrc`):

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
```

Reload the shell: `source ~/.zshrc`

## Setup (first time only)

**1. Create `Gemfile`** in the repo root:

```ruby
source "https://rubygems.org"

gem "jekyll", "~> 4.3"
gem "webrick", "~> 1.7"
gem "minima", "~> 2.5"
gem "rouge", "~> 4.0"

group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
end

# Gems removed from Ruby stdlib in 3.4+
gem "csv"
gem "bigdecimal"
gem "base64"
gem "logger"
gem "ostruct"
```

**2. Install gems:**

```bash
bundle install
```

**3. Patch Liquid** — Liquid 4.0.3 calls `String#tainted?` which was removed in Ruby 3.2. Patch it once after install:

```bash
sed -i '' 's/return unless obj\.tainted?/return unless obj.respond_to?(:tainted?) \&\& obj.tainted?/' $(bundle show liquid)/lib/liquid/variable.rb
```

This edits the installed gem source directly and persists as long as the gem version doesn't change.

## Serving the site

```bash
bundle exec jekyll serve --detach
```

The site will be available at **http://127.0.0.1:4000/**.

To stop the server:

```bash
pkill -f jekyll
```

## Troubleshooting

**`Address already in use` on port 4000** — a previous Jekyll process is still running. Kill it first:

```bash
pkill -f jekyll
bundle exec jekyll serve --detach
```

**`cannot load such file` errors on startup** — a stdlib gem is missing. Add it to the `Gemfile` (e.g. `gem "csv"`) and re-run `bundle install`.

**Sass `@import` deprecation warnings** — suppress them by adding `quiet_deps` and `silence_deprecations` to the `sass` section of `_config.yml`:

```yaml
sass:
  style: :expanded
  quiet_deps: true
  silence_deprecations: ['import']
```

**Posts rendering without styles** (plain unstyled text) — the `defaults:` block is missing from `_config.yml`. Add it:

```yaml
defaults:
  - scope:
      path: ""
      type: "posts"
    values:
      layout: "post"
  - scope:
      path: ""
      type: "pages"
    values:
      layout: "page"
```

Then restart the server.
