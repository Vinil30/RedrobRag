import jsonlines


class DescriptionLoader:

    def __init__(self):
        pass

    def load(
        self,
        file_path,
        candidate_pool
    ):

        with jsonlines.open(file_path) as reader:

            for candidate in reader:

                candidate_id = candidate["candidate_id"]

                if candidate_id not in candidate_pool:
                    continue

                candidate_pool[candidate_id]["raw_candidate"] = candidate

                description_parts = []

                profile = candidate.get("profile", {})

                if profile.get("headline"):
                    description_parts.append(profile["headline"])

                if profile.get("summary"):
                    description_parts.append(profile["summary"])

                for experience in candidate.get("career_history", []):

                    company = experience.get("company", "")
                    title = experience.get("title", "")
                    industry = experience.get("industry", "")
                    job_desc = experience.get("description", "")

                    description_parts.append(
                        f"{title} at {company} ({industry})"
                    )

                    if job_desc:
                        description_parts.append(job_desc)

                skills = candidate.get("skills", [])

                if skills:
                    skill_string = ", ".join(
                        skill["name"]
                        for skill in skills
                    )

                    description_parts.append(
                        f"Skills: {skill_string}"
                    )

                candidate_pool[candidate_id]["description"] = "\n\n".join(
                    description_parts
                )

        return candidate_pool
