import scrapy
from loguru import logger
from scrapy import signals


class LlamaBlogSpider(scrapy.Spider):
    name = "llama_blog"
    start_urls = ["https://www.llamaindex.ai/blog"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blog_count = 0
        self.total_content_length = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(LlamaBlogSpider, cls).from_crawler(crawler, *args, **kwargs)
        spider.crawler = crawler
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def parse(self, response):
        # Extract blog post elements
        blog_posts = response.css("div[class^='CardBlog_card__']")

        # Extract links and dates from the blog post elements
        for post in blog_posts:
            link = post.css("p[class*='CardBlog_title__'] a::attr(href)").get()
            date = post.css("p:last-child::text").get()
            if link:
                full_link = response.urljoin(link)
                logger.opt(colors=True).info(
                    f"Found article link: <blue>{full_link}</blue> with date: {date}"
                )
                yield response.follow(
                    full_link, self.parse_blog, meta={"date": date, "url": full_link}
                )

        # Update and log the number of blogs found on the main page
        self.blog_count = len(blog_posts)
        self.crawler.stats.set_value("blog_count", self.blog_count)
        logger.info(f"Number of blog posts found on the main page: {self.blog_count}")

    def parse_blog(self, response):
        # Extract title and content
        title = response.css("h1[class*='BlogPost_title__']::text").get()
        content_root = response.css("div[class^='BlogPost_htmlPost__']")
        # Extract all text content within content_root
        content = content_root.css("*::text").getall()

        # Combine the extracted texts into a single string
        full_content = " ".join(content).strip()

        # Extract tags from BlogPost_tags element if it exists
        tags = response.css("ul[class^='BlogPost_tags__'] li a span::text").getall()
        tags_text = ", ".join(tags)

        # Retrieve the date and URL from meta data
        date = response.meta.get("date")
        url = response.meta.get("url")

        # Extract author from the detailed blog page
        author = response.css("p[class*='BlogPost_date__'] a::text").get()

        # Log the title, content length, author, date, and tags for sanity check
        content_length = len(full_content)
        logger.info(
            f"Parsed blog post: {title} (Content length: {content_length} characters, Author: {author}, Date: {date}, Tags: {tags_text}, URL: {url})"
        )

        # Update total content length in custom stats
        self.total_content_length += content_length
        self.crawler.stats.set_value("total_content_length", self.total_content_length)

        yield {
            "title": title,
            "content": full_content,
            "author": author,
            "date": date,
            "tags": tags,
            "url": url,
        }

    def spider_closed(self, spider):
        # Log the final custom stats
        blog_count = self.crawler.stats.get_value("blog_count")
        total_content_length = self.crawler.stats.get_value("total_content_length")
        logger.info(f"Spider closed: {spider.name}")
        logger.info(f"Total number of blog posts parsed: {blog_count}")
        logger.info(
            f"Total content length of all blog posts: {total_content_length} characters"
        )
