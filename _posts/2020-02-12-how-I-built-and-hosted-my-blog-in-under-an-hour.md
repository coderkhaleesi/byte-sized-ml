---
layout: post
title: "How I built and hosted my blog in under 1 hour?"
subtitle: "Using Github pages and Jekyll"
date: 2020-02-12 01:58:13 -0400
background: '/img/posts/06.jpg'
---

<p>One particular day, as in today, I was feeling very unproductive. I couldn't study much of a research paper I had planned to study today and was really frustrated at myself. After talking to a fellow Master's student, who gave me the idea of starting a blog for writing technical posts, I decided to try it out.</p>

<p>Using blogspot.com was not an option as I read several articles detailing why it was bad. Also, I had previously used for a personal blog of mine, and there were many weird visits (bots I guess) which was skewing the traffic of the website. So I decided to go with a static generator like Jekyll and a simple hosting - Github pages (which is free of cost as well). Jekyll being a static page generator, doesn't process any PHP (cannot), which makes my website pretty secure and I don't have to worry about any PHP/database security vulnerabilities. Also, it's fast as it doesn't use any database (no database calls at every request).</p>

<h2 class="section-heading">Steps:</h2>

<p>I first installed Ruby using a simple Windows installer for Ruby</p>
<p>The next step was to install jekyll and bundler : gem install jekyll bundler</p>
<p>Check if jekyll is installed using : jekyll -v</p>
<p>I created a directory called blog: mkdir blog</p>
<p>Now before we proceed, I did not want to create a jekyll website from scratch. So what I did was I forked this repository: <a href="https://github.com/BlackrockDigital/startbootstrap-clean-blog-jekyll">Bootstrap Clean Blog Jekyll</a> into my github and I cloned it into my "blog" directory on local machine. I changed some important configurations by editing the _config.yml file:

<ul>
title:              Byte-Sized ML
<br>
email:              tanya2911dixit@gmail.com
<br>
description:        A blog dedicated to all things ML
<br>
author:             Tanya Dixit
<br>
baseurl:            "/byte-sized-ml"
<br>
url:                "https://coderkhaleesi.github.io"
</ul>
</p>
<p>Since I did not want to use the forked "BlackrockDigital" repository, I created my own repo on github. The steps were:

<ul>
    <li>Click "New Repository"</li>
    <li>Name the repository same as your baseurl in the _config.yml file. Name it so that when you access your webpage it will be in the form : url/baseurl </li>
    <li>Create a new branch in this repo called "gh-pages". Make this the default branch by going in the repo settings</li>
    <li>clone this repo in another folder on your local machine, let's say "final-blog" and copy all the files from the "blog" folder to this folder</li>

    I know I took the longer route, but I was trying this for the first time. If you want, you can directly download the files from the bootstrap theme into this repository.
    <li>Run "bundle install"</li>
    <li>Run "bundle exec jekyll serve". You will get something like this:</li>


<br>
<br>

Configuration file: C:/Users/hp/byte-sized-ml/byte-sized-ml/_config.yml
<br>
            Source: C:/Users/hp/byte-sized-ml/byte-sized-ml
<br>
       Destination: C:/Users/hp/byte-sized-ml/byte-sized-ml/_site
<br>
 Incremental build: disabled. Enable with --incremental
 <br>
      Generating...
       Jekyll Feed: Generating feed for posts
                           done in 17.17 seconds.
<br>
 Auto-regeneration: enabled for 'C:/Users/hp/byte-sized-ml/byte-sized-ml'
 <br>
    Server address: http://127.0.0.1:4000/byte-sized-ml/
    <br>
  Server running... press ctrl-c to stop.
  <br>

</ul>
Open "http://127.0.0.1:4000/byte-sized-ml/" in your browser. Your website should be ready.

</p>

<p>Uh uh uh...!!! It's still not up. Just push to your repo that you created following the steps above (don't forget to make gh-pages as the default branch) and then navigate to url/baseurl. Github will host your website.</p>

<

<blockquote class="blockquote">Happy Coding!!!!</blockquote>


<h4>References</h4>
<ul>
    <li>https://idratherbewriting.com/documentation-theme-jekyll/mydoc_publishing_github_pages.html</li>
    <li>https://learn.cloudcannon.com/jekyll/why-use-a-static-site-generator/</li>
    <li>https://blog.webjeda.com/why-jekyll-over-wordpress/</li>
    <li>https://blog.webjeda.com/jekyll-ssl/</li>
    <li>https://github.com/BlackrockDigital/startbootstrap-clean-blog-jekyll</li>
</ul>

<p>Placeholder text by <a href="http://spaceipsum.com/">Space Ipsum</a>. Photographs by <a href="https://unsplash.com/">Unsplash</a>.</p>
