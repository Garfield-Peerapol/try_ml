import googleapiclient.discovery
import re
from youtube_comment_downloader import YoutubeCommentDownloader
import time

YOUTUBE_API_KEY = "AIzaSyDw32OerQjphNXAUpV9Z-WHFPlQ437YvUI"  # แนะนำให้อ่านจาก .env จริง ๆ

def get_video_id(url):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:m\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=|embed\/|v\/|)([\w-]{11})(?:\S+)?"
    match = re.match(regex, url)
    if match:
        return match.group(1)
    return None

def get_video_info(video_id):
    youtube_api = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube_api.videos().list(part="snippet,statistics", id=video_id)
    response = request.execute()
    if response and response['items']:
        item = response['items'][0]
        return {
            "title": item['snippet']['title'],
            "viewCount": int(item['statistics'].get('viewCount', 0)),
            "likeCount": int(item['statistics'].get('likeCount', 0)),
            "commentCount": int(item['statistics'].get('commentCount', 0))
        }
    return None

def get_comments_with_downloader(video_id, max_retries=3, progress_callback=None):
    downloader = YoutubeCommentDownloader()
    comments_data = []
    attempt = 0
    while attempt < max_retries:
        try:
            comments = downloader.get_comments(video_id)
            for i, comment in enumerate(comments):
                comment_info = {
                    "id": comment.get('cid'),
                    "text": comment.get('text'),
                    "author": comment.get('author'),
                    "timestamp": comment.get('time_parsed'),
                    "likes": comment.get('votes'),
                    "isReply": comment.get('is_reply', False),
                    "parentId": comment.get('parent'),
                    "replies": []
                }
                comments_data.append(comment_info)

                if progress_callback and i % 10 == 0:
                    progress_callback(i)

            return comments_data
        except Exception as e:
            attempt += 1
            time.sleep(2)
            if attempt == max_retries:
                raise e

def organize_comments(comments_list):
    comments_by_id = {c['id']: c for c in comments_list}
    organized_comments = []
    for comment in comments_list:
        if comment['isReply'] and comment['parentId'] in comments_by_id:
            comments_by_id[comment['parentId']]['replies'].append(comment)
        elif not comment['isReply']:
            organized_comments.append(comment)
            organized_comments[-1]['replies'].sort(key=lambda x: x.get('timestamp', 0))
    return organized_comments

def scrape_comments(url, progress_callback=None):
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    video_info = get_video_info(video_id)
    if not video_info:
        raise ValueError("ไม่พบข้อมูลวิดีโอ")

    raw_comments = get_comments_with_downloader(video_id, progress_callback=progress_callback)
    organized_comments = organize_comments(raw_comments)

    return {
        "videoId": video_id,
        "videoInfo": video_info,
        "comments": organized_comments
    }
